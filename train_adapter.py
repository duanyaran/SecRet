import os
import random
import sys
import glob
import json
import csv
# === 关键修复：补全 typing 导入 ===
from typing import List, Dict, Any, Optional 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# 复用 ColBERT 目录下的隐私工具
CURRENT_DIR = os.path.dirname(__file__)
COLBERT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../shared_folder_colbertscifact"))
if COLBERT_DIR not in sys.path:
    sys.path.append(COLBERT_DIR)

try:
    from privacy_perturbation import (
        decompose_embeddings_into_proto_residual,
        compute_semantic_subspace_from_weight,
    )
except ImportError:
    pass

# === Adapter 保存路径 ===
ADAPTER_CKPT_PATH = "/root/data_dir/tasb_msmarco_adapter.pt"

# ========= 敏感词检测 =========
try:
    import spacy
    _spacy_nlp = None
    _spacy_available = True
except ImportError:
    _spacy_available = False
    _spacy_nlp = None

import re

def _ensure_spacy_nlp():
    global _spacy_nlp, _spacy_available
    if not _spacy_available: return None
    if _spacy_nlp is not None: return _spacy_nlp
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
        return _spacy_nlp
    except Exception:
        _spacy_available = False
        return None

def _detect_sensitive_spans(text: str):
    nlp = _ensure_spacy_nlp()
    spans = []
    if nlp is not None and text:
        doc = nlp(text)
        ignore_words = {"King", "Queen", "Thursday", "Monday", "December", "January"}
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "LOC", "GPE", "ORG", "DATE"}:
                if ent.text in ignore_words: continue
                spans.append((ent.start_char, ent.end_char))
    phone_regex = r"\b\d{11}\b"
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    for m in re.finditer(phone_regex, text): spans.append((m.start(), m.end()))
    for m in re.finditer(email_regex, text): spans.append((m.start(), m.end()))
    return list(set(spans))

def build_sensitive_mask_for_query(query_text, input_ids, attention_mask, tokenizer, force_random_ratio=0.3):
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    assert batch_size == 1
    spans = _detect_sensitive_spans(query_text)
    enc = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=128, padding=True, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = enc["offset_mapping"][0]
    if offset_mapping.size(0) > seq_len: offset_mapping = offset_mapping[:seq_len]
    sensitive_mask = torch.zeros(seq_len, dtype=torch.bool)
    entity_found = False
    if spans:
        for tid, (start, end) in enumerate(offset_mapping.tolist()):
            if tid >= seq_len: break
            if attention_mask[0, tid].item() == 0 or end <= start: continue
            for s_start, s_end in spans:
                if not (end <= s_start or start >= s_end):
                    sensitive_mask[tid] = True
                    entity_found = True
                    break
    if not entity_found and force_random_ratio > 0:
        valid_len = attention_mask.sum().item()
        if valid_len > 0:
            indices = torch.randperm(valid_len)
            num_mask = max(1, int(valid_len * force_random_ratio))
            sensitive_mask[indices[:num_mask]] = True
    return sensitive_mask.unsqueeze(0).to(device)

# ========= 本地 MS MARCO Arrow 加载器 =========

class LocalMSMARCODataset(IterableDataset):
    def __init__(self, data_dir, num_samples=50000):
        print(f"[MSMARCO] Searching for .arrow files in {data_dir}...")
        arrow_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # 只加载包含 'train' 的 arrow 文件
                if file.endswith(".arrow") and "train" in file:
                    arrow_files.append(os.path.join(root, file))
        
        if not arrow_files:
            raise FileNotFoundError(f"No train .arrow files found in {data_dir}")
        
        print(f"[MSMARCO] Found {len(arrow_files)} training files. Loading via HuggingFace Datasets...")
        self.dataset = load_dataset("arrow", data_files=arrow_files, split="train", streaming=True)
        self.num_samples = num_samples

    def __iter__(self):
        count = 0
        for sample in self.dataset:
            if count >= self.num_samples:
                break
            
            try:
                query = sample.get("query", "")
                passages = sample.get("passages", {})
                texts = passages.get("passage_text", [])
                labels = passages.get("is_selected", [])
                
                pos_indices = [i for i, label in enumerate(labels) if label == 1]
                neg_indices = [i for i, label in enumerate(labels) if label == 0]
                
                if not pos_indices or not neg_indices:
                    continue 
                
                p_text = texts[random.choice(pos_indices)]
                n_text = texts[random.choice(neg_indices)]
                
                yield (query, p_text, n_text)
                count += 1
                
            except Exception:
                continue

    def __len__(self):
        return self.num_samples

# ========= Model Classes =========
class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
    def forward(self, residual: torch.Tensor, sensitive_mask: torch.Tensor | None = None) -> torch.Tensor:
        if sensitive_mask is not None:
            down = self.down(residual)
            down = self.dropout(down)
            up = self.up(down)
            return residual + up * sensitive_mask.unsqueeze(-1).float()
        return residual + self.up(self.dropout(self.down(residual)))

class TASBRetrieverWithAdapter(nn.Module):
    def __init__(self, model_name: str, device: str = "cuda:0", adapter_rank: int = 64, adapter_ckpt_path: str = None):
        super().__init__()
        self.device = torch.device(device)
        print(f"[TAS-B] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.base_model.eval()
        with torch.no_grad():
            emb_layer = self.base_model.get_input_embeddings()
            self.semantic_subspace = compute_semantic_subspace_from_weight(emb_layer.weight)
        self.adapter = LoRAAdapter(self.base_model.config.hidden_size, rank=adapter_rank).to(self.device)
        if adapter_ckpt_path and os.path.exists(adapter_ckpt_path):
            self.adapter.load_state_dict(torch.load(adapter_ckpt_path, map_location=self.device))

    def forward_tasb(self, last_hidden_state, attention_mask=None):
        return last_hidden_state[:, 0, :] 

    def encode_document_batch(self, doc_texts: List[str], batch_size: int) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True).to(self.device)
            with torch.no_grad():
                out = self.base_model(**inputs)
                all_embs.append(self.forward_tasb(out.last_hidden_state))
        return torch.cat(all_embs, dim=0)

    def encode_query(self, query_text: str) -> torch.Tensor:
        device = self.device
        inputs = self.tokenizer(query_text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        sens_mask = build_sensitive_mask_for_query(query_text, inputs["input_ids"], inputs["attention_mask"], self.tokenizer).to(device)
        with torch.no_grad():
            e_clean = self.base_model.get_input_embeddings()(inputs["input_ids"])
        proto, res = decompose_embeddings_into_proto_residual(e_clean, self.semantic_subspace)
        pert = proto + self.adapter(res, sensitive_mask=sens_mask)
        with torch.no_grad():
            out = self.base_model(inputs_embeds=pert, attention_mask=inputs["attention_mask"])
            q_vec = self.forward_tasb(out.last_hidden_state)
        return q_vec.squeeze(0)

def compute_tasb_scores(q, d): return (q * d).sum(dim=-1)

# ====== 训练循环 ======
def train_tasb_adapter(
    data_path: str,
    model_name: str,
    num_steps: int = 20000, 
    batch_size: int = 32, 
    lr: float = 5e-4,
    kl_weight: float = 1.0,
    l2_weight: float = 0.01,
    anti_nn_weight: float = 0.1,
    cosine_push_weight: float = 1.0, 
):
    # 使用本地 Arrow 加载器
    train_dataset = LocalMSMARCODataset(data_dir=data_path, num_samples=num_steps*batch_size)
    loader = DataLoader(train_dataset, batch_size=batch_size)
    
    device = "cuda:0"
    model = TASBRetrieverWithAdapter(model_name, device=device, adapter_ckpt_path=None)
    model.adapter.train()
    
    optim = torch.optim.AdamW(model.adapter.parameters(), lr=lr)
    ce_kl = nn.KLDivLoss(reduction="batchmean")
    vocab_norm = F.normalize(model.base_model.get_input_embeddings().weight.detach(), dim=-1).to(device)

    losses = {"kl": 0.0, "l2": 0.0, "cos": 0.0, "anti": 0.0, "total": 0.0}
    pbar = tqdm(loader, desc=f"Training on MS MARCO")
    
    step = 0
    for q_txt, p_txt, n_txt in pbar:
        step += 1
        q_txt, p_txt, n_txt = list(q_txt), list(p_txt), list(n_txt)
        
        with torch.no_grad():
            p_emb = model.encode_document_batch(p_txt, batch_size=len(p_txt))
            n_emb = model.encode_document_batch(n_txt, batch_size=len(n_txt))
        
        inputs = model.tokenizer(q_txt, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        input_ids = inputs["input_ids"]
        
        masks = [build_sensitive_mask_for_query(t, input_ids[i:i+1], inputs["attention_mask"][i:i+1], model.tokenizer) for i, t in enumerate(q_txt)]
        mask_batch = torch.stack(masks).squeeze(1)

        with torch.no_grad():
            e_clean = model.base_model.get_input_embeddings()(input_ids)
        proto, res = decompose_embeddings_into_proto_residual(e_clean, model.semantic_subspace)
        res_p = model.adapter(res, sensitive_mask=mask_batch)
        e_pert = proto + res_p
        
        with torch.no_grad():
            out_clean = model.base_model(inputs_embeds=e_clean, attention_mask=inputs["attention_mask"])
            q_clean = model.forward_tasb(out_clean.last_hidden_state, inputs["attention_mask"])
        
        out_pert = model.base_model(inputs_embeds=e_pert, attention_mask=inputs["attention_mask"])
        q_pert = model.forward_tasb(out_pert.last_hidden_state, inputs["attention_mask"])
        
        score_pos_t, score_neg_t = compute_tasb_scores(q_clean, p_emb), compute_tasb_scores(q_clean, n_emb)
        score_pos_s, score_neg_s = compute_tasb_scores(q_pert, p_emb), compute_tasb_scores(q_pert, n_emb)
        
        kl = ce_kl(F.log_softmax(torch.stack([score_pos_s, score_neg_s], 1), dim=-1), 
                   F.softmax(torch.stack([score_pos_t, score_neg_t], 1), dim=-1))
        
        cos_loss = torch.tensor(0.0, device=device)
        anti_val = torch.tensor(0.0, device=device)
        
        valid = mask_batch.bool() & inputs["attention_mask"].bool()
        if valid.any():
            raw_v = e_clean[valid]
            pert_v = e_pert[valid]
            cos_loss = F.cosine_similarity(raw_v, pert_v).abs().mean()
            
            flat_ids = input_ids[valid]
            flat_pert = F.normalize(pert_v, dim=-1)
            sims = torch.matmul(flat_pert, vocab_norm.t())
            anti_val = sims.gather(1, flat_ids.unsqueeze(-1)).squeeze(-1).abs().mean()

        loss = kl_weight * kl + l2_weight * (e_pert - e_clean).pow(2).mean() + cosine_push_weight * cos_loss + anti_nn_weight * anti_val
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        pbar.set_postfix({
            "KL": f"{kl.item():.4f}", 
            "Cos": f"{cos_loss.item():.2f}", 
            "Anti": f"{anti_val.item():.2f}"
        })
        
        if step >= num_steps: break

    os.makedirs(os.path.dirname(ADAPTER_CKPT_PATH), exist_ok=True)
    torch.save(model.adapter.state_dict(), ADAPTER_CKPT_PATH)
    print("Saved.")

def find_model_snapshot(base_path: str) -> str:
    if os.path.exists(os.path.join(base_path, "config.json")): return base_path
    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subdirs = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if subdirs: return subdirs[0]
    return base_path

def main():
    # === 本地 MS MARCO 路径 ===
    msmarco_data_path = "/root/data_dir/LLM_finetune/data/msmarco"
    
    # === TAS-B 模型路径 ===
    model_base_path = "/root/data_dir/models--sentence-transformers--msmarco-distilbert-base-tas-b"
    tasb_model_path = find_model_snapshot(model_base_path)
    
    print(f"TAS-B Path: {tasb_model_path}")
    
    train_tasb_adapter(
        data_path=msmarco_data_path,
        model_name=tasb_model_path,
        num_steps=10000, 
        batch_size=64, 
        lr=5e-4,
        kl_weight=1.0, 
        l2_weight=0.01,
        anti_nn_weight=0.1, 
        cosine_push_weight=1.0 
    )

if __name__ == "__main__":
    main()