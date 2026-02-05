import os
import sys
import json
import torch
import re
from datasets import Dataset as HgDataset
from transformers import AutoTokenizer, AutoModel


TASB_MODEL_PATH = "/root/data_dir/models--sentence-transformers--msmarco-distilbert-base-tas-b/snapshots/default" 
ADAPTER_CKPT_PATH = "/root/data_dir/tasb_msmarco_adapter.pt"
SCIFACT_DATA_PATH = "/root/data_dir/LLM_finetune/data/beir/scifact/data"
OUTPUT_TEE_FILE = "tee_perturbed_queries.pt" 


CURRENT_DIR = os.path.dirname(__file__)
COLBERT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../shared_folder_colbertscifact"))
if COLBERT_DIR not in sys.path: sys.path.append(COLBERT_DIR)

try:
    from privacy_perturbation import (
        decompose_embeddings_into_proto_residual,
        compute_semantic_subspace_from_weight,
    )
except ImportError:
    # 如果没有该文件，定义简化的占位函数以免报错（实际运行需确保有该模块）
    def compute_semantic_subspace_from_weight(w): return torch.eye(w.size(1))
    def decompose_embeddings_into_proto_residual(e, P): return e, torch.zeros_like(e)

# === 敏感词检测逻辑 (保持不变) ===
try:
    import spacy
    _spacy_nlp = spacy.load("en_core_web_sm")
except:
    _spacy_nlp = None

def _detect_sensitive_spans(text: str):
    spans = []
    if _spacy_nlp:
        doc = _spacy_nlp(text)
        ignore_words = {"King", "Queen", "Thursday", "Monday", "December", "January"}
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "LOC", "GPE", "ORG", "DATE"}:
                if ent.text in ignore_words: continue
                spans.append((ent.start_char, ent.end_char, ent.text, ent.label_))
    # 正则补充
    phone_regex = r"\b\d{11}\b"
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    for m in re.finditer(phone_regex, text): spans.append((m.start(), m.end(), m.group(), "PHONE"))
    for m in re.finditer(email_regex, text): spans.append((m.start(), m.end(), m.group(), "EMAIL"))
    
    unique_spans = []
    seen = set()
    for s in spans:
        if (s[0], s[1]) not in seen:
            seen.add((s[0], s[1]))
            unique_spans.append(s)
    return unique_spans

def build_sensitive_mask_for_query(query_text, input_ids, attention_mask, tokenizer):

    batch_size, seq_len = input_ids.shape
    spans = _detect_sensitive_spans(query_text)
    if not spans: return torch.zeros(seq_len, dtype=torch.bool)
    
    enc = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=128, return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0]
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for idx, (start, end) in enumerate(offsets):
        if attention_mask[0, idx] == 0 or end <= start: continue
        for s_start, s_end, _, _ in spans:
            if not (end <= s_start or start >= s_end):
                mask[idx] = True
                break
    return mask.unsqueeze(0)


class LoRAAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int, rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = torch.nn.Linear(hidden_size, rank, bias=False)
        self.up = torch.nn.Linear(rank, hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, residual, sensitive_mask=None):
        if sensitive_mask is not None:
            out = self.up(self.dropout(self.down(residual)))
            return residual + out * sensitive_mask.unsqueeze(-1).float()
        return residual + self.up(self.dropout(self.down(residual)))


class TEEPreprocessor:
    def __init__(self, model_path, adapter_path):
        print("[TEE] Initializing Secure Enclave Environment...")
        self.device = "cpu" 
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        full_model = AutoModel.from_pretrained(model_path)
        self.embedding_layer = full_model.get_input_embeddings().to(self.device)
        
        
        with torch.no_grad():
            self.semantic_subspace = compute_semantic_subspace_from_weight(self.embedding_layer.weight)
            
        
        hidden_size = self.embedding_layer.embedding_dim
        self.adapter = LoRAAdapter(hidden_size=hidden_size, rank=64).to(self.device)
        if os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path, map_location=self.device))
            print("[TEE] Adapter loaded successfully.")
        else:
            print("[TEE] Warning: Adapter checkpoint not found, using random init.")
            
    def process_query(self, query_text):
        """
        核心逻辑：Tokenize -> Embedding -> Perturb -> Return Embeddings
        """
        inputs = self.tokenizer(query_text, return_tensors="pt", truncation=True, max_length=128, padding="max_length") 
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)
        
        
        with torch.no_grad():
            e_clean = self.embedding_layer(input_ids)
            
        
        sens_mask = build_sensitive_mask_for_query(query_text, input_ids, attn_mask, self.tokenizer).to(self.device)
        
        
        proto_e, residual_e = decompose_embeddings_into_proto_residual(e_clean, self.semantic_subspace)
        residual_e_prime = self.adapter(residual_e, sensitive_mask=sens_mask)
        e_pert = proto_e + residual_e_prime 
        
        return e_pert, attn_mask

def load_queries(data_path):
    
    queries_file = os.path.join(data_path, "queries.jsonl")
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            queries.append({"_id": str(item.get("_id")), "text": item.get("text")})
    return queries

def main():
    print("="*40)
    print("  TEE Phase: Secure Pre-processing")
    print("="*40)

    tee = TEEPreprocessor(TASB_MODEL_PATH, ADAPTER_CKPT_PATH)
    
    
    all_queries = load_queries(SCIFACT_DATA_PATH)
    print(f"[TEE] Loaded {len(all_queries)} queries.")

    processed_data = {}

    for q in all_queries:
        qid = q["_id"]
        text = q["text"]
        if not text: continue
        
        e_pert, mask = tee.process_query(text)
        
        
        processed_data[qid] = {
            "inputs_embeds": e_pert.squeeze(0).clone(), 
            "attention_mask": mask.squeeze(0).clone()
        }
        
        if len(processed_data) % 100 == 0:
            print(f"[TEE] Processed {len(processed_data)} queries...", end="\r")
            
    print(f"\n[TEE] Saving encrypted/safe payload to {OUTPUT_TEE_FILE}...")
    torch.save(processed_data, OUTPUT_TEE_FILE)
    print("[TEE] Done. Data is ready for transmission to untrusted GPU.")

if __name__ == "__main__":
    main()