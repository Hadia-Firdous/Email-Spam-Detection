import streamlit as st
import joblib as jl

model = jl.load("model.pkl")
vectorizer = jl.load("vectorizer.pkl")

st.title("Email Spam Detector")

st.write("Enter an an email to check if it is spam or not")

user_email = st.text_area("Enter the email you want to verify")

if st.button("Predict"):

  if user_email.strip() == "":
    st.warning("Enter an email first")

  else:
      user_email = vectorizer.transform([user_email])

      prediction = model.predict(user_email)[0]

      probabilities = model.predict_proba(user_email)

      spam = probabilities[0][1] * 100
      ham = probabilities[0][0] * 100

      if(prediction == 1):
        st.error("This email is spam")
        st.write("Spam Probability: ", round(spam,2), "%")

      else:
        st.success("This email is not spam")  
        st.write("Not-spam Probability: ", round(ham,2), "%")