\# 🫁 Pneumonia Detection from Chest X-ray



A deep learning web app that detects \*\*Pneumonia\*\* from chest X-ray images using a fine-tuned ResNet18 model.



This project demonstrates an end-to-end machine learning pipeline:

\- Data loading

\- Model training

\- Evaluation

\- Deployment using Streamlit



\## 🚀 Features



\- Upload chest X-ray images

\- Predicts \*\*Normal\*\* or \*\*Pneumonia\*\*

\- Real-time inference

\- Simple and clean UI using Streamlit



\## 🧠 Model Details



\- Model: ResNet18 (pretrained on ImageNet)

\- Transfer Learning used

\- Final layer modified for 2 classes

\- Loss Function: CrossEntropyLoss

\- Optimizer: Adam



\## 📊 Performance



| Metric | Value |

|--------|------|

| Accuracy | \~95% |

| Pneumonia Recall | \~96% |

| False Negatives | 32 |



> In medical applications, recall is critical to reduce missed pneumonia cases.



\## 🖥️ Demo



!\[App Screenshot](screenshot.png)



\## ⚙️ Installation



```bash

git clone https://github.com/Abhhiiissshhek/pneumonia-detection.git

cd pneumonia-detection

pip install -r requirements.txt

streamlit run app.py





\---



\# 📁 STEP 8 — PROJECT STRUCTURE



```markdown

\## 📂 Project Structure



pneumonia-detection/

│

├── app.py

├── src/

├── requirements.txt

├── README.md



\## 📁 Dataset



Dataset used:

\- Chest X-ray dataset from Kaggle



(Note: Dataset not included due to size)



\## ⚠️ Note



\- Model file (`.pth`) is not included due to size limits

\- You can train the model using `train.py`



\## 🚀 Future Improvements



\- Reduce false negatives

\- Add confidence score output

\- Deploy app online

\- Improve generalization



\## 👨‍💻 Author



Abhishek Prajapati (abhhiiissshhek\_ml)

