from keras.models import load_model


model=load_model('classifyHuman.h5')

def predict(img):
    score=model.predict(img)
    if score>0.5:
        return 1
    else:
        return 0