import pickle


with open("results/test_image/000.pkl", 'rb') as f:
    data = pickle.load(f)
print(data)