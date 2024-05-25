from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np

app = FastAPI()

app.mount("/statics",StaticFiles(directory="./statics"),name="statics")
# Load the model
model = pickle.load(open('model.pkl', 'rb'))

templates = Jinja2Templates(directory="./templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, feature1: int = Form(...), feature2: int = Form(...)):
#     # Add as many features as needed in the function parameters
#     int_features = [feature1, feature2]  # Add other features to the list
#     final = [np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction = model.predict_proba(final)
#     output = '{0:.{1}f}'.format(prediction[0][1], 2)

#     if float(output) > 0.5:
#         prediction_text = f'You need a treatment.\nProbability of mental illness is {output}'
#     else:
#         prediction_text = f'You do not need treatment.\nProbability of mental illness is {output}'
    
#     return templates.TemplateResponse("index.html", {"request": request, "pred": prediction_text})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, Age: int = Form(...), Gender: int = Form(...), Family_history: int = Form(...),Benefits:int=Form(...),Care_options:int=Form(...),Anonymity:int=Form(...),Leave:int=Form(...),Work_interfere:int=Form(...)):
    # Ensure the features match what your model expects
    int_features = [Age, Gender, Family_history,Benefits,Care_options,Anonymity,Leave,Work_interfere]  # Update the list to include all features
    final = [np.array(int_features)]
    
    # Debugging prints
    print("Received features:", int_features)
    print("Final array for prediction:", final)
    
    # Ensure the model is loaded if uncommented
    prediction = model.predict_proba(final)
    # Mock prediction for testing purposes
    # prediction = [[0.1, 0.9]]  # Replace this line with the actual prediction
    print("prediction=",prediction)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print("Output=",output)

    if float(output) > 0.5:
        prediction_text = f'You need a treatment.\nProbability of mental illness is {output}'
    else:
        prediction_text = f'You do not need treatment.\nProbability of mental illness is {output}'
    print("prediction_text=",prediction_text)
    return templates.TemplateResponse("index.html", {"request": request, "pred": prediction_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
