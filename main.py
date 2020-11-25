import tkinter as tk
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSE = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

f = open('covid-19.txt', 'r', errors='ignore')
raw = f.read()

raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
lem = nltk.stem.WordNetLemmatizer()
remove_pun_dict = dict((ord(pun), None) for pun in string.punctuation)

# region Window App
app = tk.Tk()
app.title('Clippo - Covid-19 ChatBot')
app.geometry('600x300')
app.configure(background='LightSkyBlue1')
# endregion


def chatBot():
    user_response = InputQuestion.get()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            T.config(state='normal')
            T.delete(1.0, tk.END)
            T.insert(tk.END, "You are welcome! Take care...")
            T.config(state='disabled')
        else:
            if greeting(user_response) is not None:
                T.config(state='normal')
                T.delete(1.0, tk.END)
                T.insert(tk.END, greeting(user_response))
                T.config(state='disabled')
            else:
                T.config(state='normal')
                T.delete(1.0, tk.END)
                T.insert(tk.END, response(user_response) + "\nWhat else can I do for you?")
                T.config(state='disabled')
                sent_tokens.remove(user_response)
    else:
        T.config(state='normal')
        T.delete(1.0, tk.END)
        T.insert(tk.END, "See you soon! Take care...")
        T.config(state='disabled')


# region GUI
LabelRow = tk.Label(app, text='ME: ', heigh='3', width='10', bg='LightSkyBlue1', anchor="e")
LabelRow.grid(row=1, column=1)
InputQuestion = tk.Entry(app, width=80)
InputQuestion.grid(row=1, column=2)

LabelRow = tk.Label(app, text='ROBOT: ', heigh='1', width='10', bg='LightSkyBlue1', anchor="e")
LabelRow.grid(row=2, column=1, sticky='N'+'E')
T = tk.Text(app, height=10, width=60)
T.grid(row=2, column=2)

LabelRow = tk.Label(app, text=' ', heigh='1', width='10', bg='LightSkyBlue1', anchor="e")
LabelRow.grid(row=3, column=1)

SendMessageButton = tk.Button(app, text='SEND', command=chatBot, width=15, heigh=2)
SendMessageButton.grid(row=4, column=2, sticky='E'+'S')
# endregion


def LemTokens(tokens):
    return [lem.lemmatize(token) for token in tokens]


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_pun_dict)))


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSE)


def response(response_toUser):
    robot_response = ''
    sent_tokens.append(response_toUser)

    tfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidfVec.fit_transform(sent_tokens)
    values = cosine_similarity(tfidf[-1], tfidf)
    idx = values.argsort()[0][-2]
    flat = values.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    count = 0
    for item in flat:
        if item > 0:
            count = count + 1

    if req_tfidf == 0:
        robot_response = robot_response + "I am sorry, I don't understand you *sad face*"
        return robot_response
    elif req_tfidf > 0.27:
        robot_response = robot_response + sent_tokens[idx]
        return robot_response
    else:
        if count >= 5:
            robot_response = robot_response + sent_tokens[idx]
            return robot_response
        else:
            robot_response = robot_response + "I am sorry, I don't have information about that."
            return robot_response


T.insert(tk.END, "Hello, my name is Clippo.\nI will answer all your questions about Covid-19.")
T.config(state='disabled')
app.mainloop()
