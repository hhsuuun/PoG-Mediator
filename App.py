import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy 
from sqlalchemy import func             
import dynamic_predict_llm as dpllm
import webbrowser
from threading import Timer, Thread
from deep_translator import GoogleTranslator

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "build")

EN_TO_ZH = {
    # Main categories
    "Civil": "民事", 
    "Criminal": "刑事",
    
    # Case classifications (Sub Category) 
    "Land & Building": "土地建物",
    "Disturbance of Peace": "安寧",
    "Traffic Accident": "車禍",
    "Burglary": "侵入竊盜",
    "Infringement": "侵害",
    "Labor Dispute": "勞資糾紛",
    "Intellectual Property": "智慧財產",
    "Fraud": "詐欺",
    "Sales & Lease": "買賣租賃",
    "Defamation": "誹謗",
    "Debt": "債務",
    "Inheritance": "遺產",
    "Injury": "傷害",
    "Medical Dispute": "醫療",
    "Relationship Dispute": "感情糾紛",
    "Alimony": "贍養",
    "Property Damage": "毀損",
    "Harassment": "騷擾"
}
app = Flask(__name__, static_folder=BUILD_DIR, static_url_path="")
CORS(app)

# ==========================================
# Connect to MySQL in Docker
# ==========================================
DB_USER = os.environ.get("DB_USER", "mediation_user")
DB_PASS = os.environ.get("DB_PASS", "securepassword")
DB_HOST = os.environ.get("DB_HOST", "localhost") 
DB_NAME = os.environ.get("DB_NAME", "mediation_db")

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database table structure
class CaseRecord(db.Model):
    __tablename__ = 'case_records'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    main_type = db.Column(db.String(50))
    sub_type = db.Column(db.String(50))
    days = db.Column(db.Integer)
    rounds = db.Column(db.Integer)
    people = db.Column(db.Integer)
    content = db.Column(db.Text)
    keywords = db.Column(db.Text)
    prediction = db.Column(db.String(20))
    explanation = db.Column(db.Text)
    paths = db.Column(db.Text)

with app.app_context():
    db.create_all()


_ai_initialized = False
_ai_initializing = False
_ai_init_error = None

def _preload_model():
    global _ai_initialized, _ai_initializing, _ai_init_error
    _ai_initializing = True
    print(">>> Background preloading AI model...")
    try:
        dpllm.initialize()
        _ai_initialized = True
        print(">>> AI model preload complete.")
    except Exception as e:
        _ai_init_error = str(e)
        print(f">>> AI model preload failed: {e}")
    finally:
        _ai_initializing = False

def ensure_ai_initialized():
    import time
    if _ai_initialized:
        return
    print(">>> Waiting for background preload to complete...")
    while _ai_initializing:
        time.sleep(1)
    if not _ai_initialized:
        raise RuntimeError(f"AI model loading failed: {_ai_init_error}")

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"ai_ready": _ai_initialized, "ai_loading": _ai_initializing})

@app.route("/api/predict", methods=["POST"])
def predict():
    """Write-time translation"""
    data = request.json
    user_text = data.get("text", "").strip()
    
    # 1. Get English from frontend, convert to Chinese for AI use
    en_main_type = data.get("main_type", "Civil")
    en_sub_type = data.get("sub_type", "Traffic Accident")
    
    zh_main_type = EN_TO_ZH.get(en_main_type, "民事")
    zh_sub_type = EN_TO_ZH.get(en_sub_type, en_sub_type) # if not found, keep original
    
    days = data.get("days", 30)
    rounds = data.get("rounds", 1)
    people = data.get("people", 2)

    if not user_text:
        return jsonify({"error": "Content cannot be empty"}), 400

    try:
        # 2. Check if the exact same case already exists in database
        existing_record = CaseRecord.query.filter_by(content=user_text).first()
        if existing_record:
            # Split the paths saved in the database with newline separator(\n) back into an array for frontend
            saved_paths = existing_record.paths.split('\n') if existing_record.paths else []
            return jsonify({
                "source": "database",
                "keywords": existing_record.keywords,
                "prediction": existing_record.prediction,
                "paths": saved_paths if saved_paths else ["No paths found in historical database."],
                "explanation": f"[Fetched from DB Cache] {existing_record.explanation}",
                "db_id": existing_record.id
            })

        # 3. Call AI for inference (using Chinese labels)
        ensure_ai_initialized()
        zh_keywords = dpllm.extract_keywords(user_text, dpllm.few_shot_examples)
        
        case_data = {
            "案件大類型": zh_main_type, "案件分類": zh_sub_type, "調解天數": days,
            "調解次數": rounds, "調解人數": people, "text": user_text, "keywords": zh_keywords
        }

        zh_pred_str = dpllm.predict_ensemble(case_data)
        df_row_for_kg = {"案件分類": zh_sub_type, "案件細節類型": "", "關鍵詞": zh_keywords, "預測結果": zh_pred_str}
        zh_paths = dpllm.find_subgraph_paths_from_case(df_row_for_kg, dpllm.kg, max_paths=5, cutoff=6)
        zh_paths_str = [" -> ".join(p) for p in zh_paths]
        zh_explanation = dpllm.summarize_and_infer(zh_main_type, zh_sub_type, zh_keywords, zh_paths)

        # 4. Translate AI results to English
        translator = GoogleTranslator(source='zh-TW', target='en')
        try:
            en_keywords = translator.translate(zh_keywords)
            en_explanation = translator.translate(zh_explanation)
            en_paths_str = [translator.translate(p) for p in zh_paths_str]
        except:
            en_keywords = zh_keywords
            en_explanation = zh_explanation
            en_paths_str = zh_paths_str
            
        en_pred_str = "Successful" if zh_pred_str == "成立" else "Unsuccessful"

        # 5. Save all English data to MySQL
        joined_paths_for_db = '\n'.join(en_paths_str) if 'en_paths_str' in locals() else '\n'.join(zh_paths_str)
        new_record = CaseRecord(
            main_type=en_main_type, sub_type=en_sub_type, days=days, rounds=rounds, people=people,
            content=user_text, keywords=en_keywords, prediction=en_pred_str, explanation=en_explanation,
            paths=joined_paths_for_db  # [New] Write paths to MySQL
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            "source": "ai",
            "keywords": en_keywords,
            "prediction": en_pred_str,
            "paths": en_paths_str,
            "explanation": en_explanation,
            "db_id": new_record.id
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"AI Error: {e}"}), 500

# ==========================================
# Dashboard API
# ==========================================
@app.route("/api/stats", methods=["GET"])
def stats_api():
    try:
        total_cases = CaseRecord.query.count()
        success_cases = CaseRecord.query.filter_by(prediction="Successful").count()
        avg_days = db.session.query(func.avg(CaseRecord.days)).scalar()
        
        type_stats = db.session.query(
            CaseRecord.sub_type,
            func.count(CaseRecord.id).label('total'),
            func.sum(func.if_(CaseRecord.prediction == 'Successful', 1, 0)).label('success')
        ).group_by(CaseRecord.sub_type).all()
        chart_data = [
            {"name": row.sub_type, "Total Cases": row.total, "Success Cases": int(row.success) if row.success else 0}
            for row in type_stats
        ]

        return jsonify({
            "total_cases": total_cases,
            "success_rate": round((success_cases / total_cases) * 100, 1) if total_cases > 0 else 0,
            "avg_days": round(float(avg_days), 1) if avg_days else 0,
            "chart_data": chart_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/cases", methods=["GET"])
def search_cases():
    try:
        keyword = request.args.get("keyword", "").strip()
        prediction = request.args.get("prediction", "")
        main_type = request.args.get("main_type", "")
        sub_type = request.args.get("sub_type", "")
        
        query = CaseRecord.query
        
        if keyword:
            query = query.filter((CaseRecord.content.contains(keyword)) | (CaseRecord.keywords.contains(keyword)))
        if prediction:
            query = query.filter(CaseRecord.prediction == prediction)
        if sub_type:
            query = query.filter(CaseRecord.sub_type == sub_type)
        if main_type:
            query = query.filter(CaseRecord.main_type == main_type)
            
        cases = query.order_by(CaseRecord.id.desc()).limit(100).all()
        
        return jsonify([{
            "id": c.id, "main_type": c.main_type, "sub_type": c.sub_type,
            "days": c.days, "rounds": c.rounds, "people": c.people,
            "prediction": c.prediction, "keywords": c.keywords,
            "content": (c.content[:100] + "...") if c.content and len(c.content) > 100 else (c.content or ""),
            "explanation": c.explanation
        } for c in cases])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    full_path = os.path.join(BUILD_DIR, path)
    if path and os.path.exists(full_path):
        return send_from_directory(BUILD_DIR, path)
    return send_from_directory(BUILD_DIR, "index.html")

def open_browser():
    webbrowser.open_new("http://localhost:5001")

if __name__ == "__main__":
    Thread(target=_preload_model, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, threaded=True)