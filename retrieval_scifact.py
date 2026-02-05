import os
import sys
import json
import csv
from typing import Dict, Any, List

import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import util, models, SentenceTransformer
from datasets import Dataset as HgDataset

# Reuse RoBERTa/ColBERT implementation for privacy perturbation
CURRENT_DIR = os.path.dirname(__file__)
COLBERT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../shared_folder_colbertscifact"))
if COLBERT_DIR not in sys.path:
    sys.path.append(COLBERT_DIR)
try:
    from privacy_perturbation import (
        mutual_information_kl_np,
        decompose_embeddings_into_proto_residual,
        compute_semantic_subspace_from_weight,
    )
except ImportError:
    pass

# === 修改 1: Adapter 路径改为 TAS-B + SciFact ===
ADAPTER_CKPT_PATH = "/root/data_dir/tasb_msmarco_adapter.pt"

# ====== Sensitive Token Detection (Unchanged) ======
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
                spans.append((ent.start_char, ent.end_char, ent.text, ent.label_))

    phone_regex = r"\b\d{11}\b"
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    student_id_regex = r"\b\d{8,12}\b"

    for m in re.finditer(phone_regex, text):
        spans.append((m.start(), m.end(), m.group(), "PHONE"))
    for m in re.finditer(email_regex, text):
        spans.append((m.start(), m.end(), m.group(), "EMAIL"))
    for m in re.finditer(student_id_regex, text):
        spans.append((m.start(), m.end(), m.group(), "STUDENT_ID"))

    seen = set()
    unique_spans = []
    for s in spans:
        key = (s[0], s[1], s[3])
        if key not in seen:
            seen.add(key)
            unique_spans.append(s)
    return unique_spans

def build_sensitive_mask_for_query(
    query_text: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    assert batch_size == 1
    spans = _detect_sensitive_spans(query_text)
    if not spans:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    enc = tokenizer(
        query_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
        return_offsets_mapping=True,
    )
    offset_mapping = enc["offset_mapping"][0]

    sensitive_mask = torch.zeros(seq_len, dtype=torch.bool)
    span_intervals = [(s[0], s[1]) for s in spans]

    for tid, (start, end) in enumerate(offset_mapping.tolist()):
        if attention_mask[0, tid].item() == 0: continue
        if end <= start: continue
        for s_start, s_end in span_intervals:
            if not (end <= s_start or start >= s_end):
                sensitive_mask[tid] = True
                break
    return sensitive_mask.unsqueeze(0).to(device)


# ====== Data Loader (SciFact 适配) ======
def get_doc_text(doc_item):
    title = doc_item.get("title", "")
    text = doc_item.get("text", "")
    # SciFact 适配
    if not text:
        text = doc_item.get("abstract", "")
        if isinstance(text, list):
            text = " ".join(text)
    return (str(title) + " " + str(text)).strip()

def load_scifact_data(data_path: str):
    parent_dir = os.path.dirname(data_path.rstrip("/"))
    corpus_file = os.path.join(data_path, "corpus.jsonl")
    if not os.path.exists(corpus_file): corpus_file = os.path.join(parent_dir, "corpus.jsonl")
    if not os.path.exists(corpus_file): raise FileNotFoundError(f"未找到 corpus.jsonl: {data_path}")

    print("[Loader] Loading SciFact Corpus...")
    corpus_list = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["_id"] = str(item.get("_id", item.get("doc_id", item.get("id"))))
                item["content"] = get_doc_text(item)
                corpus_list.append(item)
    corpus = HgDataset.from_list(corpus_list)

    queries_file = os.path.join(data_path, "queries.jsonl")
    queries_list = []
    qrels_list = []
    query_split = {}

    if os.path.exists(queries_file):
        print("[Loader] Loading BEIR format Queries...")
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    queries_list.append({
                        "_id": str(item.get("_id", item.get("id"))),
                        "text": str(item.get("text", ""))
                    })
        qrels_dir = os.path.join(data_path, "qrels")
        splits = ["train", "dev", "test"]
        if not os.path.exists(os.path.join(qrels_dir, "dev.tsv")) and os.path.exists(os.path.join(qrels_dir, "validation.tsv")):
            splits = ["train", "validation", "test"]
        for split in splits:
            file_path = os.path.join(qrels_dir, f"{split}.tsv")
            if os.path.exists(file_path):
                print(f"  - Loading {split} qrels from {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    next(reader, None)
                    for row in reader:
                        if len(row) < 3: continue
                        qid, docid, score = row[0], row[1], int(row[2])
                        query_split[str(qid)] = split
                        if score > 0: qrels_list.append({"query-id": str(qid), "corpus-id": str(docid), "label": 1})
    else:
        print("[Loader] Loading Raw SciFact Claims...")
        def load_claims(fname, split_name):
            fpath = os.path.join(data_path, fname)
            if not os.path.exists(fpath): fpath = os.path.join(parent_dir, fname)
            if not os.path.exists(fpath): return
            print(f"  - Loading {fname}...")
            with open(fpath, "r") as f:
                for line in f:
                    item = json.loads(line)
                    qid = str(item.get("id"))
                    text = item.get("claim", "")
                    queries_list.append({"_id": qid, "text": text})
                    query_split[qid] = split_name
                    for doc_id, ev in item.get("evidence", {}).items():
                        qrels_list.append({"query-id": qid, "corpus-id": str(doc_id), "label": 1})
        
        load_claims("claims_train.jsonl", "train")
        load_claims("claims_dev.jsonl", "dev")
        load_claims("claims_test.jsonl", "test")

    return corpus, HgDataset.from_list(queries_list), qrels_list, query_split

# ====== Adapter ======
class LoRAAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int, rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = torch.nn.Linear(hidden_size, rank, bias=False)
        self.up = torch.nn.Linear(rank, hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, residual: torch.Tensor, sensitive_mask: torch.Tensor | None = None) -> torch.Tensor:
        if sensitive_mask is not None:
            down_proj = self.down(residual)
            down_proj = self.dropout(down_proj)
            up_proj = self.up(down_proj)
            sensitive_mask_expanded = sensitive_mask.unsqueeze(-1).float()
            return residual + up_proj * sensitive_mask_expanded
        else:
            return residual + self.up(self.dropout(self.down(residual)))

# ====== CORE CLASS: TAS-B + Adapter + Forced CLS Pooling ======

class TASBRetrieverWithAdapterForEval:
    def __init__(
        self,
        query_model_name: str,
        doc_model_name: str,
        device: str = "cuda:0",
        adapter_rank: int = 64,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # === 核心修改: 手动构建 CLS Pooling 模型 ===
        def build_cls_model(model_path):
            print(f"[Init] Building model from {model_path} with FORCED CLS Pooling...")
            word_emb = models.Transformer(model_path)
            pooling = models.Pooling(
                word_emb.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=False, 
                pooling_mode_cls_token=True,    # 关键: 开启 CLS
                pooling_mode_max_tokens=False
            )
            return SentenceTransformer(modules=[word_emb, pooling], device=str(self.device))

        # 1. 加载 Document Encoder
        self.doc_model = build_cls_model(doc_model_name)
        self.doc_model.eval()

        # 2. 加载 Query Base Model
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.base_model = AutoModel.from_pretrained(query_model_name).to(self.device)
        self.base_model.eval()

        # 3. 计算子空间
        with torch.no_grad():
            emb_layer = self.base_model.get_input_embeddings()
            hidden_size = emb_layer.embedding_dim
            self.semantic_subspace = compute_semantic_subspace_from_weight(emb_layer.weight)

        # 4. 加载 Adapter
        self.adapter = LoRAAdapter(hidden_size=hidden_size, rank=adapter_rank).to(self.device)

        if os.path.exists(ADAPTER_CKPT_PATH):
            state = torch.load(ADAPTER_CKPT_PATH, map_location=self.device)
            self.adapter.load_state_dict(state)
            print(f"[TAS-B Adapter] 已从 {ADAPTER_CKPT_PATH} 加载预训练权重")
        else:
            print(f"[TAS-B Adapter] 警告: 未找到 {ADAPTER_CKPT_PATH}，将使用随机初始化")

    def encode_corpus(self, corpus_texts: List[str], batch_size: int = 32) -> torch.Tensor:
        print(f"Encoding corpus ({len(corpus_texts)} documents) with CLS Pooling...")
        with torch.no_grad():
            # TAS-B 也是基于 Dot Product 的，为了稳健性开启 Normalize
            return self.doc_model.encode(
                corpus_texts, 
                convert_to_tensor=True, 
                batch_size=batch_size, 
                show_progress_bar=True,
                normalize_embeddings=True 
            )

    def encode_query(self, query_text: str) -> torch.Tensor:
        """
        单条 Query 编码 (带敏感词扰动)
        """
        device = self.device
        inputs = self.tokenizer(
            query_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        # Mask 生成
        sensitive_mask = build_sensitive_mask_for_query(
            query_text=query_text,
            input_ids=input_ids,
            attention_mask=attn_mask,
            tokenizer=self.tokenizer,
        ).to(device)

        # Embedding 提取
        embedding_layer = self.base_model.get_input_embeddings()
        with torch.no_grad():
            e_clean = embedding_layer(input_ids)

        # 扰动逻辑
        proto_e, residual_e = decompose_embeddings_into_proto_residual(
            e_clean, self.semantic_subspace
        )
        residual_e_prime = self.adapter(residual_e, sensitive_mask=sensitive_mask)
        e_pert = proto_e + residual_e_prime

        # Forward Pass
        with torch.no_grad():
            model_outputs = self.base_model(
                inputs_embeds=e_pert,
                attention_mask=attn_mask,
            )
            last_hidden_state = model_outputs.last_hidden_state

        # === 手动 CLS Pooling ===
        q_vec = last_hidden_state[:, 0, :]
        
        # === 关键修正：归一化 ===
        q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=-1)
        
        return q_vec.squeeze(0)

# ====== Metrics Eval (Unchanged) ======
def evaluate_retrieval_metrics_optimized(
    retriever: TASBRetrieverWithAdapterForEval,
    corpus: HgDataset,
    queries: HgDataset,
    qrels_list: List[Dict],
    query_split: Dict[str, str],
    split: str = "test",
    k_list: List[int] = [50, 100],
    batch_size: int = 32
):
    print(f"\n开始评估 Split: {split}")
    target_qids = {qid for qid, sp in query_split.items() if sp == split}
    q_text_map = {str(item["_id"]): item["text"] for item in queries if str(item["_id"]) in target_qids}
    
    qrels_map = {}
    for qrel in qrels_list:
        qid, doc_id = str(qrel["query-id"]), str(qrel["corpus-id"])
        if qid in target_qids: qrels_map.setdefault(qid, set()).add(doc_id)
    
    eval_qids = [qid for qid in q_text_map.keys() if qid in qrels_map]
    eval_queries_text = [q_text_map[qid] for qid in eval_qids]
    
    print(f"有效评估 Query 数量: {len(eval_qids)}")
    if len(eval_qids) == 0: return {}

    # 1. Encode Corpus
    corpus_items = list(corpus)
    corpus_texts = [item["content"] for item in corpus_items]
    corpus_ids = [str(item["_id"]) for item in corpus_items]
    
    doc_embeddings = retriever.encode_corpus(corpus_texts, batch_size=batch_size)

    # 2. Encode Queries
    print("Encoding queries (with perturbation)...")
    query_vecs = []
    for q_text in tqdm(eval_queries_text, desc="Encoding Queries"):
        vec = retriever.encode_query(q_text) 
        query_vecs.append(vec.cpu()) 
    
    query_embeddings = torch.stack(query_vecs).to(retriever.device) 

    # 3. Retrieve
    max_k = max(k_list)
    print(f"Retrieving top-{max_k}...")
    
    # Dot Product
    hits = util.semantic_search(
        query_embeddings, 
        doc_embeddings, 
        top_k=max_k,
        score_function=util.dot_score
    )

    # 4. Calc Metrics
    metrics = {k: {"recall": [], "precision": [], "f1": []} for k in k_list}

    for i, query_hits in enumerate(hits):
        qid = eval_qids[i]
        true_doc_ids = qrels_map[qid]
        retrieved_doc_ids = [corpus_ids[hit['corpus_id']] for hit in query_hits]

        for k in k_list:
            retrieved_k = set(retrieved_doc_ids[:k])
            num_hits = len(retrieved_k & true_doc_ids)
            num_rel = len(true_doc_ids)
            recall = num_hits / num_rel if num_rel > 0 else 0.0
            precision = num_hits / k
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            metrics[k]["recall"].append(recall)
            metrics[k]["precision"].append(precision)
            metrics[k]["f1"].append(f1)

    final_results = {}
    for k in k_list:
        final_results[f"recall@{k}"] = float(np.mean(metrics[k]["recall"]))
        final_results[f"precision@{k}"] = float(np.mean(metrics[k]["precision"]))
        final_results[f"f1@{k}"] = float(np.mean(metrics[k]["f1"]))
        
    return final_results

# ====== Privacy Eval (Unchanged) ======
def collect_sensitive_embeddings(
    retriever: TASBRetrieverWithAdapterForEval,
    queries,
    query_split: Dict[str, str],
    split: str = "test",
    max_queries: int = 200,
    max_tokens: int = 5000,
):
    query_text_by_id = {str(q["_id"]): q["text"] for q in queries}
    target_ids = [qid for qid, sp in query_split.items() if sp == split]
    if not target_ids and split == "test":
        target_ids = [qid for qid, sp in query_split.items() if sp == "dev"]

    collected_clean = []
    collected_pert = []
    collected_token_ids = []
    device = retriever.device
    embedding_layer = retriever.base_model.get_input_embeddings()

    import random
    random.shuffle(target_ids)
    count_q = 0

    for qid in target_ids:
        if count_q >= max_queries: break
        text = query_text_by_id.get(str(qid))
        if not text: continue

        inputs = retriever.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        with torch.no_grad():
            e_clean = embedding_layer(input_ids)

        sens_mask = build_sensitive_mask_for_query(text, input_ids, attn_mask, retriever.tokenizer).to(device)
        
        proto_e, residual_e = decompose_embeddings_into_proto_residual(e_clean, retriever.semantic_subspace)
        residual_e_prime = retriever.adapter(residual_e, sensitive_mask=sens_mask)
        e_pert = proto_e + residual_e_prime

        valid_mask = attn_mask.bool() & sens_mask.bool()
        if not valid_mask.any(): continue

        collected_clean.append(e_clean[valid_mask].cpu())
        collected_pert.append(e_pert[valid_mask].cpu())
        collected_token_ids.append(input_ids[valid_mask].cpu())

        count_q += 1
        if sum(v.size(0) for v in collected_clean) >= max_tokens: break

    if not collected_clean: return None, None, None
    return torch.cat(collected_clean, 0), torch.cat(collected_pert, 0), torch.cat(collected_token_ids, 0)

def evaluate_nn_mi_cosine_privacy(retriever, queries, query_split, split="test", max_queries=200, max_tokens=5000):
    clean_emb, pert_emb, token_ids = collect_sensitive_embeddings(retriever, queries, query_split, split, max_queries, max_tokens)
    if clean_emb is None: return

    device = retriever.device
    print(f"\n[PrivacyEval] 收集敏感 token: N={clean_emb.size(0)}")

    with torch.no_grad():
        emb_weight = retriever.base_model.get_input_embeddings().weight.to(device)
        vocab_norm = torch.nn.functional.normalize(emb_weight, dim=-1)
        clean_norm = torch.nn.functional.normalize(clean_emb.to(device), dim=-1)
        pert_norm = torch.nn.functional.normalize(pert_emb.to(device), dim=-1)

        inv_acc_clean = (clean_norm @ vocab_norm.t()).argmax(dim=-1).cpu().eq(token_ids).float().mean().item()
        inv_acc_pert = (pert_norm @ vocab_norm.t()).argmax(dim=-1).cpu().eq(token_ids).float().mean().item()

    cos = torch.nn.functional.cosine_similarity(clean_norm, pert_norm, dim=-1).abs()

    clean_np = clean_emb.detach().cpu().numpy()
    pert_np = pert_emb.detach().cpu().numpy()
    try:
        mi_approx = mutual_information_kl_np(clean_np, pert_np, k=5)
    except Exception as e:
        print(f"MI calc failed: {e}")
        mi_approx = 0.0
    
    print("-" * 40)
    print(f"[TAS-B Privacy] Split: {split}")
    print(f"  Inversion Acc (Clean -> Pert): {inv_acc_clean:.4f} -> {inv_acc_pert:.4f}")
    print(f"  Cosine Similarity (Mean):      {cos.mean().item():.4f}")
    print(f"  Mutual Information (Approx):   {mi_approx:.4f}")
    print("-" * 40 + "\n")

# ====== Helper function ======
def find_model_snapshot(base_path: str) -> str:
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path
    
    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subdirs = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if subdirs:
            print(f"Using snapshot: {subdirs[0]}")
            return subdirs[0]
    return base_path

def main():
    # === 修改 2: 指向 SciFact ===
    scifact_data_path = "/root/data_dir/LLM_finetune/data/beir/scifact/data"
    
    model_base_path = "/root/data_dir/models--sentence-transformers--msmarco-distilbert-base-tas-b"
    tasb_model_path = find_model_snapshot(model_base_path)
    
    print(f"TAS-B Path: {tasb_model_path}")
    
    print("=" * 80)
    print("TAS-B + Adapter (SciFact) | Correct Normalization")
    print("=" * 80)

    corpus, queries, qrels_list, query_split = load_scifact_data(scifact_data_path)
    
    retriever = TASBRetrieverWithAdapterForEval(
        query_model_name=tasb_model_path, 
        doc_model_name=tasb_model_path, 
        adapter_rank=64
    )

    # SciFact 通常用 dev 作为验证集，Test 集无标签
    eval_split = "dev"
    if eval_split not in query_split.values(): 
        print("Warning: dev split not found, check dataset loader.")

    # 1. Retrieval
    metrics = evaluate_retrieval_metrics_optimized(
        retriever, corpus, queries, qrels_list, query_split, 
        split=eval_split, k_list=[50, 100], batch_size=128
    )

    print("\nRetrieval Results:")
    for k in [50, 100]:
        print(f"  K={k}: R={metrics.get(f'recall@{k}',0):.4f}, P={metrics.get(f'precision@{k}',0):.4f}, F1={metrics.get(f'f1@{k}',0):.4f}")

    # 2. Privacy
    evaluate_nn_mi_cosine_privacy(
        retriever, queries, query_split, split=eval_split, max_queries=200, max_tokens=5000
    )

if __name__ == "__main__":
    main()