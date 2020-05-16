import pandas as pd #pandas is a fast data analysis and manipulation tool.

#This is an array which contains some SMS, both spam and ham. They will be used to predict the type.
#It represents the dataset.
sms_data = pd.read_csv('SMSSpamCollection', header=None, sep='\t', names=['Label', 'SMS'])

#I'm going to work on a copy of the SMS' array.

sms_data_clean = sms_data.copy()

#Get the whole sentences, remove the punctuation, convert the text into the lower-case, splitting words.
#I'm going to do these instructions because every word is indipendent from others.

sms_data_clean['SMS'] = sms_data_clean['SMS'].str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()

sms_data_clean['SMS'] = sms_data_clean['SMS'].str.lower()

sms_data_clean['SMS'] = sms_data_clean['SMS'].str.split();

#Split to train and test data.

#Train data are going to be 80% of the whole dataset chosen randomly. reset_index resets the array's index order from 0 to the final index.
train_data = sms_data_clean.sample(frac=0.8, random_state=1).reset_index(drop=True)
test_data = sms_data_clean.drop(train_data.index).reset_index(drop=True) #These are the 20% left data.
train_data = train_data.reset_index(drop=True)

#Prepare vocabulary - the list of all the words from the dataset

vocabulary = list(set(train_data['SMS'].sum()))

#Calculate frequencies of the words for each message

word_counts_per_sms = pd.DataFrame([
    [row[1].count(word) for word in vocabulary]
    for _, row in train_data.iterrows()], columns=vocabulary)

train_data = pd.concat([train_data.reset_index(), word_counts_per_sms], axis=1).iloc[:,1:]

#Calculate values for the Bayes formula.

alpha=1
Nvoc = len(train_data.columns) - 3 #The number of unique words in the whole dataset.
Pspam = train_data['Label'].value_counts()['spam'] / train_data.shape[0] #The part of spam messages in our dataset.
Pham = train_data['Label'].value_counts()['ham'] / train_data.shape[0] #The part of non-spam messages in our dataset.
Nspam = train_data.loc[train_data['Label'] == 'spam', 'SMS'].apply(len).sum() #The total number of words in the spam messages.
Nham = train_data.loc[train_data['Label'] == 'ham', 'SMS'].apply(len).sum() ##The total number of words in the non-spam messages.

#This is the formula to determine the probabilities of the given word to belong to spam. It's conditional probability.
def p_w_spam(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['Label'] == 'spam', word].sum() + alpha) / (Nspam + alpha*Nvoc)
    else:
        return 1

#This is the formula to determine the probabilities of the given word to belong to non-spam.
def p_w_ham(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['Label'] == 'ham', word].sum() + alpha) / (Nham + alpha*Nvoc)
    else:
        return 1

#Prepare the classificator

def classify(message):
    p_spam_given_message = Pspam #The probability to be a spam message.
    p_ham_given_message = Pham #The probability to be a non-spam message.
    for word in message:
        p_spam_given_message *= p_w_spam(word) #Probability that the message is spam.
        p_ham_given_message *= p_w_ham(word) #Probability that the message is non-spam.
    if p_ham_given_message > p_spam_given_message:
        return 'ham' #It's non-spam.
    elif p_ham_given_message < p_spam_given_message:
        return 'spam' #It's spam.
    else:
        return 'needs human classification' #The algorithm cannot give an answer, so it needs an human.

test_data['predicted'] = test_data['SMS'].apply(classify) #For each sentence the classify function will be called. The input parameter is the sentence itself.

print(test_data.head())


