# QA_Celebrity

- Create virtual environment
```
python3.10 -m venv myenv
source myenv/bin/activate
```

- Install dependencies
```
pip install -r requirements.txt
pip install faiss-gpu
```

- Prepare data
```
python data_prepare.py
```

- Run app
```
streamlit run streamlit_app.py --server.runOnSave false
```
