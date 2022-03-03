# οι βιβλιοθήκες string & collections είναι built_in 
# βιβλιοθήκες της python και δεν χρειάζεται να τις 
# εγκαταστήσετε χειροκίνητα
import string # θα μας βοηθήσει για την επεξεργασία κειμένου
from collections import Counter # θα μας βοηθλησει για να μετράμε τον αριθμό εμφάνισης λέξεων

# βιβλιοθήκη που θα μας βοηθήσει να παρουσιάσουμε γραφικά
# την ανάλυση μας
import matplotlib.pyplot as plt

# Διαβάζουμε το αρχείο read.txt και το αποθηκεύουμε
# σε μια μεταβλητή
# το αρχείο read.txt περιέχει μία ομιλία του Trump πριν τις εκλογές, link:
# https://edition.cnn.com/2020/08/28/politics/donald-trump-speech-transcript/index.html
text = open("read.txt", encoding="utf-8").read()

# Όλο το κείμενο το κάνουμε σε μικρά γράμματα διότι
# δε θέλουμε η λέξη love να αναγνωρίζεται ως διαφορετική
# από τη λέξη LOVE
lower_case = text.lower()

# Αφαιρούμε σύμβολα όπως !,#,?,<,> για να καθαρίσουμε το κείμενο μας
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Το επεξεργασμένο κείμενο το χωρίζουμε ανά λέξη
# για παράδειγμα τη φράση i love burgers θα μας επιστραφεί
# σαν μια λίστα ['i', 'love', 'burgers']
tokenized_words = cleaned_text.split()

# Τα srot words είναι λέξεις οι οποίος δεν δηλώνουν συναισθήματα
# θα μπορούσαμε να χρησιμοποιήσουμε τα stop words που μας προσφέρει η 
# βιβλιοθήκη nltk αλλά σε αυτό το στάδιο φτιάξαμε μία δικιά μας λίστα
# με stop words 
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# Εδώ κρατάμε όσες λέξεις δεν είναι στη λίστα stop_words
# για να έχουμε στη λίστα final_words μόνο λέξεις που δηλώνουν συναισθήματα
final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)

# το emotion_list θα είναι μια λίστα η οποία ανά λέξη
# αντιστοιχίζεται και ένα συναίσθημα, για παράδειγμα η λέξη inspired
# αντιστοιχίζεται στο συναίσθημα happy
emotion_list = []
# διαβάζουμε το αρείο emotions.txt το οποίο περιέχει ενδεικτικές
# αντιστοιχίες λέξεων με συναισθήματα
with open('emotions.txt', 'r') as file:
    # για κάθε γραμμή του αρχείου
    for line in file:
        # αφαιρούμε τον χαρακτήρα ' από τις λέξεις, για παράδειγμα η πρώτη
        # γραμμή από 'victimized': 'cheated', θα γίνει victimized: cheated
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        # για κάθε λέξη του κειμένου που υπάρχει στη λίστα με τα emotions
        # την προσθέτουμε στη λίστα emotion_list
        if word in final_words:
            emotion_list.append(emotion)

# τυπώνουμε ενδεικτικά τη λίστα
print(emotion_list)
# μετράμε πόσες φορές βρέθηκε ένα συναίσθημα στη λίστα emotion_list
w = Counter(emotion_list)
# το τυπωνουμε
print(w)

# τέλος εμφανίζουμε το γράφημα με την ανάλυση μας

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.show()