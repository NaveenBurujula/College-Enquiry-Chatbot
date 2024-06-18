import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

professor_data = {
    "E Vijaya": {"ID": "P001", "Contact": "+1 234 567 8901", "Teaches": "Data Structures, DBMS, Software Engineering",
                 "Position": "Assistant Professor"},
    "Pooja Prasad": {"ID": "P002", "Contact": "+1 234 567 8902", "Teaches": "Computer Networks",
                     "Position": "Assistant Professor"},
    "Dr.Swathi": {"ID": "P003", "Contact": "+1 234 567 8903", "Teaches": "JAVA, Data structures",
                  "Position": "Assistant Professor"},
    "Raheera": {"ID": "P004", "Contact": "+1 234 567 8904", "Teaches": "SML", "Position": "Associate Professor"},
    "Dr.Sudhakar": {"ID": "P005", "Contact": "+1 234 567 8905", "Teaches": "Machine Learning",
                    "Position": "Assistant Professor"},
}

student_data = {
    "Burujula Naveen": {"ID": "21R11A0506", "Department": "CSE", "Contact": "+91 7671800324", "CGPA": "8.46",
                        "Email": "21r11a0506@gcet.edu.in"},
    "P Reena": {"ID": "21R11A0541", "Department": "CSE", "Contact": "+91 6305124575", "CGPA": "7.5",
                "Email": "21r11a0541@gcet.edu.in"},
    "Nithin Yadav": {"ID": "21R11A0532", "Department": "CSE", "Contact": "+91 9491460899", "CGPA": "7.2",
                     "Email": "21r11a0532@gcet.edu.in"},
    "Lasya": {"ID": "21R11A0503", "Department": "CSE", "Contact": "+91 9014755380", "CGPA": "8.3",
              "Email": "21r11a0503@gcet.edu.in"},
    "L Dhanya": {"ID": "21R11A0527", "Department": "CSE", "Contact": "+91 9390905132", "CGPA": "8.1",
                 "Email": "21r11a0527@gcet.edu.in"},
}

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if "professor_name" in tag or "student_name" in tag:
                    name = extract_name_from_msg(msg)  # Assuming a function to extract name
                    if name in professor_data:
                        details = professor_data[name]
                        return f"Professor {name}, ID: {details['ID']}, Contact: {details['Contact']}, Teaches: {details['Teaches']}, Position: {details['Position']}"
                    elif name in student_data:
                        details = student_data[name]
                        return f"Student {name}, ID: {details['ID']}, Department: {details['Department']}, Contact: {details['Contact']}, CGPA: {details['CGPA']}, Email: {details['Email']}"
                    else:
                        return "Sorry, I don't have information about that person."
                return random.choice(intent['responses'])
    
    return "I do not understand..."

def extract_name_from_msg(msg):
    # Basic extraction based on input format, adjust as needed
    words = msg.split()
    for word in words:
        if word in professor_data or word in student_data:
            return word
    return None

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
