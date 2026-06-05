import os
import sys
import torch
import pandas as pd
from llama_cpp import Llama
import networkx as nx
import re
import itertools
import random
from typing import List, Tuple
import torch.nn.functional as F
import json
import joblib 

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig, 
)
from peft import PeftModel 
import subprocess
import numpy as np

MODEL_DIR = "ensemble_3path"

PT_PATH = os.path.join(MODEL_DIR, "roberta_model.pt")
ROBERTA_ID = "hfl/chinese-roberta-wwm-ext-large"

# Path 3 TAIDE constants
TAIDE_ID = "taide/Llama-3.1-TAIDE-LX-8B-Chat"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LORA_PATH = os.path.join(MODEL_DIR, "taide_lora")

SPECIAL_TOKENS = ["[時間]", "[人名]", "[車號]", "[地點]"]
MODEL_FILE = "ensemble_3path/taide-7b-a.2-q4_k_m.gguf"
KG_FILE = "ensemble_3path/demo_kg.gml"
FEW_SHOT_CSV = "files/few_shot.csv"
LABEL_MAP = {0: 0, 1: 1}
STRING_MAP = {0: "不成立", 1: "成立"}

# global variables to hold models and resources
llm = None
roberta_model = None
roberta_tokenizer = None
roberta_temperature = 1.0

taide_cls_model = None
taide_cls_tokenizer = None

xgb_model = None
xgb_ohe = None

ensemble_meta = None
kg = None
few_shot_examples = ""


def initialize(regenerate_kg=False):
    """Load all AI models and knowledge graph for API calls."""
    global llm, roberta_model, roberta_tokenizer, roberta_temperature
    global taide_cls_model, taide_cls_tokenizer
    global xgb_model, xgb_ohe, ensemble_meta, kg, few_shot_examples

    print("\n[System Startup] Starting to load AI model library...\n")

    # 1. Load generative TAIDE (GGUF) - responsible for field extraction and summarization
    if os.path.exists(MODEL_FILE):
        print(f"-> Loading generative model TAIDE-LX-7B GGUF ({MODEL_FILE})")
        llm = Llama(
            model_path=MODEL_FILE,
            n_gpu_layers=-1,
            n_ctx=2048,
            low_vram=True,
            verbose=False,
        )
    else:
        print(f"⚠️ Generative model not found: {MODEL_FILE}")

    # 2. Load Path 2: RoBERTa model and temperature
    roberta_model, roberta_tokenizer, _, temp_val = load_model_from_pt()
    roberta_temperature = float(temp_val) if temp_val else 1.0
    if roberta_model:
        print(f"-> Path 2 RoBERTa loaded successfully (calibrated temperature T={roberta_temperature:.4f})")

    # 3. Load Path 3: TAIDE 8-bit classifier (LoRA)
    try:
        print("-> Loading Path 3 TAIDE classifier (4-bit quantization + LoRA)...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            TAIDE_ID,
            num_labels=2,
            token=HF_TOKEN,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        taide_cls_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        taide_cls_model.eval()

        taide_cls_tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        print("-> Path 3 TAIDE classifier loaded successfully!")
    except Exception as e:
        print(f"⚠️ Path 3 TAIDE loading failed: {e}")

    # 4. Load Path 1: XGBoost and Ensemble recipe (updated path)
    try:
        xgb_ohe_path = os.path.join(MODEL_DIR, "xgb_ohe.joblib")
        xgb_model_path = os.path.join(MODEL_DIR, "xgb_calibrated_model.joblib")
        meta_path = os.path.join(MODEL_DIR, "meta.json")

        xgb_ohe = joblib.load(xgb_ohe_path)
        xgb_model = joblib.load(xgb_model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            ensemble_meta = json.load(f)
        print("-> Path 1 XGBoost and Meta recipe loaded successfully!")
    except Exception as e:
        print(f"⚠️ XGBoost or Meta loading failed: {e}")

    # 5. Load few-shot examples
    if os.path.exists(FEW_SHOT_CSV):
        df_fs = pd.read_csv(FEW_SHOT_CSV)
        few_shot_df = df_fs[["案件內容_匿名", "人工_關鍵詞"]].dropna()
        few_shot_examples = "\n".join(
            [
                f"Text: {row['案件內容_匿名']} -> Keywords: {row['人工_關鍵詞']}"
                for _, row in few_shot_df.iterrows()
            ]
        )
        print(f"-> Loaded {len(few_shot_df)} few-shot examples.")

    # 6. Knowledge graph initialization
    if os.path.exists(KG_FILE):
        kg = nx.read_gml(KG_FILE, label="label")
        print(f"-> Knowledge graph loaded successfully! Number of nodes: {kg.number_of_nodes()}")
    else:
        print(f"⚠️ Knowledge graph file not found: {KG_FILE}")

    print("\n[System Startup] All AI modules initialized! Waiting for requests...\n")


# Keyword tokenization and processing
def get_processed_keywords(keyword_string):
    if pd.isna(keyword_string):
        return []
    raw_kw = str(keyword_string)
    keywords = [kw.strip() for kw in re.split(r"[ ,，、；;]", raw_kw) if kw.strip()]
    return keywords


def load_model_from_pt(device="cpu"):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading prediction model to {device}...")

    try:
        ckpt = torch.load(PT_PATH, map_location=device)
    except FileNotFoundError:
        print(f"Model weight file not found: {PT_PATH}, cannot load prediction model.")
        return None, None, None, None

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_ID, use_fast=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ckpt.get("special_tokens", SPECIAL_TOKENS)}
    )

    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_ID, num_labels=2)
    model.resize_token_embeddings(ckpt.get("vocab_size", len(tokenizer)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    embedding = ckpt.get("embedding", None)
    temperature = ckpt.get("temperature", None)

    return model, tokenizer, embedding, temperature


def predict_outcome_roberta(
    text: str, model, tokenizer, label_map: dict
) -> Tuple[str, float]:
    global roberta_temperature  # Import global temperature variable

    if model is None or tokenizer is None:
        return "Model not loaded", 0.0

    if not text or not isinstance(text, str) or text.strip() == "":
        return "Text is empty", 0.0

    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply Temperature Scaling calculated during training
    if roberta_temperature != 1.0 and roberta_temperature > 0:
        logits = logits / roberta_temperature

    # Get calibrated probability (Softmax)
    probabilities = F.softmax(logits, dim=-1)

    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_probability = probabilities[0, predicted_class_idx].item()
    prediction_label_num = label_map.get(predicted_class_idx, "Unknown result")

    return prediction_label_num, predicted_probability


# Keyword extraction function
def extract_keywords(text, few_shot_examples, max_tokens=50):
    prompt = (
        f"{few_shot_examples}\n"
        f"Extract keywords from the following text based on the above examples, each keyword about 2-3 characters:\n"
        f"Text: {text}\n-> Keywords:"
    )
    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        stop=["\n", "Text:"],
        temperature=0.0,
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def predict_ensemble(case_data):
    """
    Perform weighted fusion prediction through three-path ensemble model (XGBoost + RoBERTa + TAIDE LoRA).
    Input case_data dictionary containing text and structured features from frontend.
    """
    global xgb_model, xgb_ohe, roberta_model, roberta_tokenizer, roberta_temperature
    global taide_cls_model, taide_cls_tokenizer, ensemble_meta

    print("\n--- [Three-path Ensemble Activated] Computing confidence of each path model ---")

    # ==========================================
    # Path 1: Tabular XGBoost Prediction
    # ==========================================
    prob1 = 0.5
    if xgb_model is not None and xgb_ohe is not None:
        try:
            # Create single-row DataFrame format to match OHE field requirements
            df_xgb = pd.DataFrame(
                [
                    {
                        "調解次數": case_data["調解次數"],
                        "調解人數": case_data["調解人數"],
                        "調解天數": case_data["調解天數"],
                        "案件大類型": case_data["案件大類型"],
                        "案件分類": case_data["案件分類"],
                    }
                ]
            )
            num_feats = (
                df_xgb[["調解次數", "調解人數", "調解天數"]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
            cat_feats = df_xgb[["案件大類型", "案件分類"]].astype(str).fillna("UNK")

            # Feature encoding and combination
            cat_enc = xgb_ohe.transform(cat_feats)
            X_xgb = np.hstack([num_feats.values, cat_enc])

            # Get calibrated probability of establishment (Class 1)
            prob1 = float(xgb_model.predict_proba(X_xgb)[0, 1])
            print(f"  > [Path 1] XGBoost tabular prediction probability: {prob1:.4f}")
        except Exception as e:
            print(f"  ⚠️ [Path 1] XGBoost prediction failed: {e}")

    # ==========================================
    # Path 2: RoBERTa Text Prediction (with temperature calibration)
    # ==========================================
    prob2 = 0.5
    if roberta_model is not None and roberta_tokenizer is not None:
        try:
            rob_device = next(roberta_model.parameters()).device
            inputs_rob = roberta_tokenizer(
                case_data["text"], return_tensors="pt", truncation=True, max_length=512
            ).to(rob_device)

            with torch.no_grad():
                outputs_rob = roberta_model(**inputs_rob)

            logits_rob = outputs_rob.logits
            # Apply optimal temperature T calculated during training for Softmax
            prob2 = float(
                torch.softmax(logits_rob / roberta_temperature, dim=-1)[0, 1]
                .cpu()
                .item()
            )
            print(f"  > [Path 2] RoBERTa text prediction probability: {prob2:.4f}")
        except Exception as e:
            print(f"  ⚠️ [Path 2] RoBERTa prediction failed: {e}")

    # ==========================================
    # Path 3: TAIDE (Llama-3) LoRA Classifier Prediction
    # ==========================================
    prob3 = 0.5
    if taide_cls_model is not None and taide_cls_tokenizer is not None:
        try:
            # Reconstruct Prompt prefix structure consistent with training phase
            prefix = f"Classification:[{case_data['案件大類型']}/{case_data['案件分類']}] Mediation:[{case_data['調解次數']} times/{case_data['調解人數']} people/{case_data['調解天數']} days] Content:"
            text_taide_inference = f"According to statistics, the establishment rate of civil cases is about 50%; the establishment rate of criminal cases is about 50%.\n{prefix}\n{case_data['text']}"

            tai_device = next(taide_cls_model.parameters()).device
            inputs_tai = taide_cls_tokenizer(
                text_taide_inference,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(tai_device)

            with torch.no_grad():
                outputs_tai = taide_cls_model(**inputs_tai)

            # Apply Softmax directly to classification model score output
            prob3 = float(torch.softmax(outputs_tai.logits, dim=-1)[0, 1].cpu().item())
            print(f"  > [Path 3] TAIDE LoRA classification prediction probability: {prob3:.4f}")
        except Exception as e:
            print(f"  ⚠️ [Path 3] TAIDE classification prediction failed: {e}")

    # ==========================================
    # Ultimate Three-Path Weighted Fusion (Ensemble Decision)
    # ==========================================
    if ensemble_meta is not None:
        w = ensemble_meta["weights"]  # Format: [w1, w2, w3]
        threshold = ensemble_meta["threshold"]

        # Weight according to golden ratio formula
        final_prob = (w[0] * prob1) + (w[1] * prob2) + (w[2] * prob3)
        pred_str = "成立" if final_prob >= threshold else "不成立"

        print(
            f"\n[Ensemble Result] Final weighted probability: {final_prob:.4f} (Decision threshold: {threshold:.3f})"
        )
        print(f"[Ensemble Result] System final determination: 【{pred_str}】\n")
        return pred_str
    else:
        # Fallback: if formula cannot be read, use voting method for safety
        print("  ⚠️ meta.json formula not found, using default equal-weight decision.")
        final_prob = (prob1 + prob2 + prob3) / 3.0
        return "成立" if final_prob >= 0.5 else "不成立"


# Fuzzy match nodes and path finding function
def fuzzy_match_node(kg, keyword):
    keyword = keyword.strip()
    full_entity_label = f"ENTITY_{keyword}"
    if kg.has_node(full_entity_label):
        return full_entity_label

    # Next, try fuzzy matching other nodes, excluding CASE_ nodes
    matches = [
        n for n in kg.nodes if keyword in str(n) and not str(n).startswith("CASE_")
    ]
    if matches:
        # Prioritize returning Category or Detail type nodes
        category_matches = [n for n in matches if str(n).startswith("CATEGORY_")]
        detail_matches = [n for n in matches if str(n).startswith("DETAIL_")]
        if category_matches:
            return category_matches[0]
        if detail_matches:
            return detail_matches[0]
        return matches[0]
    return None


def find_subgraph_paths_from_case(
    df_row,
    kg,
    max_paths=5,
    min_length=5,
    cutoff=6,
    max_paths_per_pair=15,
    max_pairs=50,
    max_keywords=5,
):
    if kg is None:
        return []

    raw_keywords = df_row.get("關鍵詞", "")
    processed_keywords = get_processed_keywords(raw_keywords)

    target_prediction = df_row.get("預測結果", "")
    if not target_prediction or target_prediction not in ["成立", "不成立"]:
        target_prediction = "Unknown"

    candidate_start_nodes = []
    for col in ["案件分類", "案件細節類型"]:
        val = df_row.get(col, "")
        node = fuzzy_match_node(kg, val)
        if node:
            candidate_start_nodes.append(node)

    if processed_keywords:
        # Limit keyword count to avoid pair explosion
        for kw in processed_keywords[:max_keywords]:
            node = fuzzy_match_node(kg, kw)
            if node:
                candidate_start_nodes.append(node)

    candidate_start_nodes = list(set(candidate_start_nodes))
    if not candidate_start_nodes:
        return []

    all_outcome_nodes = [
        n for n, d in kg.nodes(data=True) if d.get("type", "").lower() == "outcome"
    ]

    if not all_outcome_nodes:
        return []

    if target_prediction != "未知":
        target_outcome_nodes = [
            n for n in all_outcome_nodes if n.endswith(f"_{target_prediction}")
        ]
        secondary_outcome_nodes = [
            n for n in all_outcome_nodes if not n.endswith(f"_{target_prediction}")
        ]
    else:
        target_outcome_nodes = all_outcome_nodes
        secondary_outcome_nodes = []

    def search_pairs(pairs, label):
        """Search for valid paths in the given (start, target) pair list, add to final_paths."""
        pairs_tried = 0
        for start, target in pairs:
            if len(final_paths) >= max_paths or pairs_tried >= max_pairs:
                break
            pairs_tried += 1

            target_case_id = next(
                (n for n in kg.neighbors(target) if str(n).startswith("CASE_")), None
            )
            if target_case_id in excluded_case_ids or start == target:
                continue

            try:
                # Use generator + islice instead of list(), at most max_paths_per_pair paths per pair
                path_gen = nx.all_simple_paths(
                    kg, source=start, target=target, cutoff=cutoff
                )
                shortest_path = None
                for path in itertools.islice(path_gen, max_paths_per_pair):
                    if len(path) >= min_length:
                        if shortest_path is None or len(path) < len(shortest_path):
                            shortest_path = path

                if shortest_path is None:
                    continue

                path_case_nodes = [
                    n for n in shortest_path if str(n).startswith("CASE_")
                ]
                if any(node in excluded_case_ids for node in path_case_nodes):
                    continue

                final_paths.append(shortest_path)
                if target_case_id:
                    excluded_case_ids.add(target_case_id)

            except nx.NetworkXNoPath:
                continue

    final_paths = []
    excluded_case_ids = set()

    target_pairs = list(itertools.product(candidate_start_nodes, target_outcome_nodes))
    random.shuffle(target_pairs)
    print(
        f"Prioritize searching for paths matching prediction result '{target_prediction}' (attempt at most {max_pairs} pairs)..."
    )
    search_pairs(target_pairs, "Target")

    if len(final_paths) < max_paths and secondary_outcome_nodes:
        print(f"Insufficient number of paths, supplementing with paths not matching prediction result...")
        secondary_pairs = list(
            itertools.product(candidate_start_nodes, secondary_outcome_nodes)
        )
        random.shuffle(secondary_pairs)
        search_pairs(secondary_pairs, "Secondary")

    return final_paths


def summarize_and_infer(
    case_category: str,
    case_detail: str,
    keywords: str,
    paths: List[List[str]],
    max_tokens=1000,
) -> str:
    """
    Generate a one-sentence summary and inference based on case information and KG paths.
    """
    if not paths:
        return "Unable to find related knowledge paths, cannot infer mediation result."

    # Extract outcome results from paths as reference
    path_outcomes = [path[-1] for path in paths]
    outcome_summary = "；".join(path_outcomes)

    outcome_counts = {}
    for outcome in path_outcomes:
        result = outcome.split("_")[-1]
        outcome_counts[result] = outcome_counts.get(result, 0) + 1

    majority_outcome = (
        max(outcome_counts, key=outcome_counts.get) if outcome_counts else "Unknown"
    )

    path_nodes = set(
        node
        for path in paths
        for node in path
        if not node.startswith(("CASE_", "OUTCOME_"))
        and node not in [case_category, case_detail]
    )

    relevant_disputes = [
        node.split("_")[-1] for node in path_nodes if node.split("_")[-1].strip()
    ]
    disputes_str = "、".join(list(set(relevant_disputes)))
    if not disputes_str:
        disputes_str = "No key disputes found"

    prompt = (
        f"Please act as a professional mediator and infer the final result of this mediation (established or not), and provide reasonable explanations based on the following information.\n\n"
        f"Case elements for this mediation:\n"
        f"- Case Classification: {case_category}\n"
        f"- Detail Type: {case_detail}\n"
        f"- Keywords: {keywords}\n"
        f"\n"
        f"Reference similar case outcomes: {outcome_summary}\n"
        f"Reference related paths: {disputes_str}\n"
        f"Conclusion must be consistent with {majority_outcome}\n"
        f"Inference result examples:\n"
        f"1. This case concerns property and debt issues between spouses, as debt issues are not serious so mediation is established\n"
        f"2. This case involves lease and large owed amounts, therefore mediation is not established\n\n"
        f"Please complete the following format inference in \"one sentence\":\n"
        f"Summarize the key reasons of the case, explain \"therefore infer mediation {majority_outcome}\"\n"
        f"Inference:"
    )

    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        stop=["\n", "Inference:"],
        temperature=0.0,
        echo=False,
    )
    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    initialize()

    input_file = "files/few_shot.csv"
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        exit()
    print(f"Total {len(df)} data records")

    for col in ["關鍵詞"]:
        if col not in df.columns:
            df[col] = ""

    PREDICTION_COL = "預測結果"
    if PREDICTION_COL not in df.columns:
        df[PREDICTION_COL] = pd.Series(dtype="Int64")
    elif df[PREDICTION_COL].dtype != "Int64":
        df[PREDICTION_COL] = pd.to_numeric(df[PREDICTION_COL], errors="coerce").astype(
            "Int64"
        )

    for col in ["Dynamic_KG_路徑", "Dynamic_KG_推論結果"]:
        if col not in df.columns:
            df[col] = ""
        elif df[col].dtype != "object":
            df[col] = df[col].astype(str)

    # Batch keyword extraction + KG path search + inference summary
    unlabeled_indices = df.index[1058:1060]
    batch_size = 5

    print("\nStarting keyword extraction, KG path search and inference")
    for start in range(0, len(unlabeled_indices), batch_size):
        batch_indices = unlabeled_indices[start : start + batch_size]
        print(f"\n--- Batch (indices {batch_indices[0]} to {batch_indices[-1]}) ---")

        for idx in batch_indices:
            row = df.loc[idx]
            text = row["案件內容_清理"]
            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                continue

            case_category = row.get("案件分類", "Unknown")
            case_detail = row.get("案件細節類型", "Unknown")
            print(f"\nCase Classification: {case_category} / Case Detail Type: {case_detail}")

            try:
                print(f"Processing... Record #{idx}, text length: {len(text)}")

                # Extract keywords
                raw_keywords_val = row.get("關鍵詞", "")

                if (
                    pd.isna(raw_keywords_val)
                    or str(raw_keywords_val).strip().lower() == "nan"
                    or (
                        isinstance(raw_keywords_val, str)
                        and not raw_keywords_val.strip()
                    )
                ):

                    keywords = extract_keywords(text, few_shot_examples)
                    df.loc[idx, "關鍵詞"] = keywords
                    print(f"Record #{idx} extracted new keywords: {keywords}")
                else:
                    keywords = str(raw_keywords_val).strip()
                    df.loc[idx, "關鍵詞"] = keywords
                    print(f"Record #{idx} using existing keywords: {keywords}")

                # Predict
                roberta_pred_num, roberta_prob = predict_outcome_roberta(
                    text, roberta_model, roberta_tokenizer, LABEL_MAP
                )
                roberta_pred_str = STRING_MAP.get(roberta_pred_num, "Unknown")
                df.loc[idx, "預測結果"] = roberta_pred_num
                print(f"Prediction result (numeric): {roberta_pred_num}")

                # KG search (cutoff=6, with case exclusion logic enabled)
                temp_df_row_for_kg = df.loc[idx].copy()
                temp_df_row_for_kg["預測結果"] = roberta_pred_str
                temp_df_row_for_kg["關鍵詞"] = keywords
                paths = find_subgraph_paths_from_case(
                    temp_df_row_for_kg, kg, max_paths=5, cutoff=6
                )

                # Summary and inference
                if paths:
                    print(f"Found {len(paths)} related paths:")
                    for p in paths:
                        print(" -> ".join(p))
                    df.loc[idx, "Dynamic_KG_路徑"] = " | ".join(
                        ["->".join(p) for p in paths]
                    )

                    inference = summarize_and_infer(
                        case_category, case_detail, keywords, paths
                    )
                    print(f"Inference summary: {inference}")
                    df.loc[idx, "Dynamic_KG_推論結果"] = inference

                else:
                    print("No related paths found in KG")
                    df.loc[idx, "Dynamic_KG_路徑"] = ""
                    df.loc[idx, "Dynamic_KG_推論結果"] = "KG lacks sufficient information for inference"

            except Exception as e:
                print(f"Processing record #{idx} failed! Error: {e}")
                df.loc[idx, "關鍵詞"] = ""
                df.loc[idx, "Dynamic_KG_路徑"] = ""
                df.loc[idx, "Dynamic_KG_推論結果"] = f"Processing error: {e}"

        # Save progress
        df.to_csv(input_file, index=False, encoding="utf-8")
        print(f"Progress saved to {input_file} (processed up to record #{batch_indices[-1]})")

    print("\n--- Task completed ---")
