# εισαγωγή βιβλιοθηκών
import numpy as np
import bz2
import re
import sklearn


# συνάρτηση που θα διαβάζει τα δεδομένα
# από το κείμενο
def get_labels_and_texts(file):
    #αρχικοποιούμε τις λίστες labels και text
    labels = []
    texts = []
    # για κάθε γραμμη του αρχείου που διαβάζουμε
    for line in bz2.BZ2File(file):
        # το κάνουμε decode σε utf-8
        x = line.decode("utf-8")
        # προσθέτουμε στη λίστα lables την πληροφορία
        # αν η βαθμολογία είναι αρνητική ή θετική
        labels.append(int(x[9]) - 1)
        # προσθέτουμε στη λίστα text την κριτική του χρήστη
        texts.append(x[10:].strip())

    return np.array(labels), texts

# κάνουμε επεξεργασία των αρχείων του dataset μας
train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('test.ft.txt.bz2')

# κρατάμε τα πρώτα 1000 στοιχεία για εξοικονόμηση χρόνου
train_labels=train_labels[0:1000]
train_texts=train_texts[0:1000]

def normalize_texts(texts):
    # regex καν΄όνες
    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')
    # αρχικοποίηση λίστας
    normalized_texts = []
    for text in texts:
        # μετατρέπουμε όλες τις λέξεις σε μικρά γράμματα
        # και το αποθηκεύουμε σε μια μεταβλητή
        lower = text.lower()
        # αφαιρούμε περιττά συμβολα (!,@ κτλ) από το κείμενο lower
        # και το αντικαθιστούμε με τον κενό χαρακτήρα
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        # το κανονικοποιημένο κείμενο το αποθηκεύουμε στη λιστα που
        # θα επιστρέψουμε
        normalized_texts.append(no_non_ascii)
    return normalized_texts

# εφαρμόζουμε την κανονικοποίηση στις λίστες με τα κείμενα
train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)

# κανονικοποίηση κειμένων σε μορφή πινάκων
# θέτουμε το binary = True έτσι ώστε όλες οι μη μηδενικές μετρήσεις
# να γίνονται ίσες με 1. Αυτό το κάνουμε καθώς το μοντέλο μας 
# σαν output θα μας εμφανίζει 0 ή 1.
count_vect = sklearn.feature_extraction.text.CountVectorizer(binary=True)

# εφαρμόζουμε τον count vector στα train texts
count_vect.fit(train_texts)
# στη συνέχεια τα κάνουμε transform, αυτό το κάνομε διότι μετά από
# αυτό όλα τα δεδομένα θα είναι ομοιόμορφα κατανεμημένα γύρω από τη τιμή 0 και 1.
# και η κάθε τιμή κάθε δεδομένου θα είναι μοναδικά, αυτό το κάνουμε για 
# την καλύτερη απόδοση του μοντέλου
x = count_vect.transform(train_texts)
# το εφαρμόζουμε και στα test_texts
x_test = count_vect.transform(test_texts)

# στη συνέχεια χωρίζουμε τα δεδόμενα σε train και test , το 75% των δεδομένων
# θα χρησιμοποιηθεί για την εκπαίδευση και τα υπόλοιπα 25 για το τεστάρισμα
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, train_labels, train_size = 0.75)


# Εφαρμογή αλγορίθμου logistic regression
lr = sklearn.linear_model.LogisticRegression()
# Κάνουμε train με τα δεδομένα μας
lr.fit(x_train, y_train)
# Εκτύπωση του Accuracy του μοντέλου
print (f"Accuracy: {sklearn.metrics.accuracy_score(y_val, lr.predict(x_val))}")

# θέση κριτικής που θα εξετάσουμε
p = 5
# η πρόβλεψη του μοντέλου
if lr.predict(x_test[p]) == 1:
    print("Prediction: It's a positive review")
else:
    print("Prediction: It's a negative review")

# η πραγματική τιμή
if test_labels[p] == 1:
    print("In reality: It's a positive review")
else:
    print("In reality: It's a negative review")

print("-------------")
# το κείμενο της κριτικής
print(test_texts[p])

# εξέταση σε δικιές μας κριτικές
print(lr.predict(count_vect.transform(["The new applewatch is my favourite product of apple"])))

print(lr.predict(count_vect.transform(["Samsung s6 phone is so bad, i waste my money on it"])))

print(lr.predict(count_vect.transform(["I love Rocky movies"])))