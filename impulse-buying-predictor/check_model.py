from pathlib import Path
p = Path("models/model.joblib")
print("exists:", p.exists())
print("size_MB:", round(p.stat().st_size/1024/1024, 2))
