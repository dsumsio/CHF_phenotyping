from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import ray
import string

ray.init()

df_icds_MGH_pre = pd.read_csv('MGH_ICD_all.csv')
df_icds_MGH = ray.put(df_icds_MGH_pre)
df_icds_BIDMC_pre = pd.read_csv('icd_all_with_dates.csv')
df_icds_BIDMC = ray.put(df_icds_BIDMC_pre)
df_meds_MGH_pre = pd.read_csv('MGH_MED_all.csv')
df_meds_MGH = ray.put(df_meds_MGH_pre)
df_meds_BIDMC_pre = pd.read_csv('/med_all_with_dates_update.csv')
df_meds_BIDMC = ray.put(df_meds_BIDMC_pre)
df_notes_pre1 = pd.read_csv('data_2800.csv')
length = len(df_notes_pre1)
df_notes = ray.put(df_notes_pre1)
keywords_CHF_pre = pd.read_csv('KW_CHF_BoW.txt')
notes_vocab_keywords = keywords_CHF_pre['keywords'].astype(str).tolist()
icd_CHF_pre = pd.read_csv('ICD_CHF.txt')
icds_vocab = icd_CHF_pre['ICD_cd'].astype(str).tolist()
med_CHF_pre = pd.read_csv('MED_CHF.txt')
meds_vocab = med_CHF_pre['Med_name'].astype(str).tolist()


# Initialize the stemmer
stemmer = PorterStemmer()

# set time range
time_range = pd.DateOffset(months=18)
 
def get_BoW_features_from_notes(notes, vocabulary):
    bow_vectorizer = CountVectorizer(vocabulary= vocabulary)  # create a BoW feature extractor
    bow_vectorizer.fit(notes)
    X = bow_vectorizer.transform(notes)   # get features matrix X
    X = X.toarray()    # initially X is a sparse matrix, you can convert it to a non-sparse numpy array
 
    # The resulting X is
    # array([[1, 0],    # the first note has 1 occurrence of ‘first’, and 0 occurrence of ‘second’
    #             [0, 1],    # the second note has 0 occurrence of ‘first’, and 1 occurrence of ‘second’
    #             [0, 0]],    # the third note has 0 occurrence of ‘first’, and 0 occurrence of ‘second’
    # dtype=int64)
 
    return X
    
feats = []



@ray.remote
def loop(i, df_notes, df_icds_MGH, df_icds_BIDMC, df_meds_MGH, df_meds_BIDMC, notes_vocab_keywords, icds_vocab, meds_vocab):
    # for i in tqdm(range(len(df_notes))):
        pid = df_notes.patientid.iloc[i]
        note = df_notes.note_txt.iloc[i]
        note_date = pd.to_datetime(df_notes.note_date.iloc[i])
        note_date_min = note_date - time_range
        note_date_max = note_date + time_range
 
        # this is all the ICDs this patient have close to this note
        if df_notes.hospital.iloc[i] == 'MGH': # For MGH data
            icds = df_icds_MGH.ICDcd[(df_icds_MGH.bdsppatientid==pid)&(pd.to_datetime(df_icds_MGH.contactdts)>=note_date_min) & (pd.to_datetime(df_icds_MGH.contactdts)<=note_date_max)].values
        elif df_notes.hospital.iloc[i] == 'BIDMC': # For BIDMC data
            icds = df_icds_BIDMC.diag_cd[(df_icds_BIDMC.MRN10==pid)&(pd.to_datetime(df_icds_BIDMC.adm_dt)>=note_date_min) & (pd.to_datetime(df_icds_BIDMC.adm_dt)<=note_date_max)].values
        icds = [str(x) for x in icds.tolist()]
        icds = ' '.join(icds)

        # this is all the meds this patient have close to this note
        if df_notes.hospital.iloc[i] == 'MGH':
            meds = df_meds_MGH.medicationdsc[ (df_meds_MGH.bdsppatientid == pid) &  (
                                    ((pd.to_datetime(df_meds_MGH.startdts, format='ISO8601') >= note_date_min) & (pd.to_datetime(df_meds_MGH.enddts, format='ISO8601') <= note_date_max)) | # meds start and end in window
                                    ((pd.to_datetime(df_meds_MGH.startdts, format='ISO8601') >= note_date_min) & (pd.to_datetime(df_meds_MGH.startdts, format='ISO8601') <= note_date_max) & (pd.to_datetime(df_meds_MGH.enddts, format='ISO8601') >= note_date_max)) | # start in window and end after window
                                    ((pd.to_datetime(df_meds_MGH.startdts, format='ISO8601') <= note_date_min) & (pd.to_datetime(df_meds_MGH.enddts, format='ISO8601') >= note_date_min) & (pd.to_datetime(df_meds_MGH.enddts, format='ISO8601') <= note_date_max)) | # start before window and end in window
                                    ((pd.to_datetime(df_meds_MGH.startdts, format='ISO8601') <= note_date_min) & (pd.to_datetime(df_meds_MGH.enddts, format='ISO8601') >= note_date_max)))  ].values # start before window end after window
        elif df_notes.hospital.iloc[i] == 'BIDMC':
            meds = df_meds_BIDMC.med_name[ (df_meds_BIDMC.MRN10 == pid) &  (
                                    ((pd.to_datetime(df_meds_BIDMC.start_dt) >= note_date_min) & (pd.to_datetime(df_meds_BIDMC.stop_dt) <= note_date_max)) | # meds start and end in window
                                    ((pd.to_datetime(df_meds_BIDMC.start_dt) >= note_date_min) & (pd.to_datetime(df_meds_BIDMC.start_dt) <= note_date_max) & (pd.to_datetime(df_meds_BIDMC.stop_dt) >= note_date_max)) | # start in window and end after window
                                    ((pd.to_datetime(df_meds_BIDMC.start_dt) <= note_date_min) & (pd.to_datetime(df_meds_BIDMC.stop_dt) >= note_date_min) & (pd.to_datetime(df_meds_BIDMC.stop_dt) <= note_date_max)) | # start before window and end in window
                                    ((pd.to_datetime(df_meds_BIDMC.start_dt) <= note_date_min) & (pd.to_datetime(df_meds_BIDMC.stop_dt) >= note_date_max)))  ].values # start before window end after window
        meds = [str(x) for x in meds.tolist()]
        meds = ' '.join(meds)

        # Stem the vocab words
        stemmed_words = []

        # Stem the note
        X_list = np.zeros(len(notes_vocab_keywords))
        sentences = note.split('.')
        for sentence in sentences:
            words = word_tokenize(sentence)
            stemmed_words = [stemmer.stem(word.lower()) for word in words]
            array = []
            for word in notes_vocab_keywords:
                tokens = word_tokenize(word)
                vocab = [stemmer.stem(token.lower()) for token in tokens]
                feat_notes_iter = get_BoW_features_from_notes([" ".join(stemmed_words)], vocab)
                for i in range(len(tokens)):
                    if tokens[i] == '[' or tokens[i] == ']' or tokens[i]== '(' or tokens[i]== ')':
                        result_count = sentence.count(tokens[i])
                        if result_count > 0:
                            feat_notes_iter[0,i] = result_count
                value = np.min(feat_notes_iter)
                array.append(value.astype(int))
            X_list += array
        X_list = X_list.astype(int)
        feat_notes = np.reshape(X_list, (1, len(notes_vocab_keywords)))


        # obtain the features
        feat_icds = np.array([int(re.search(r'(?:{})'.format(re.escape(x)), icds, re.IGNORECASE) is not None) for x in icds_vocab]).reshape(1,len(icds_vocab))
        feat_meds = np.array([int(re.search(r'\b{}\b'.format(re.escape(x)), meds, re.IGNORECASE) is not None) for x in meds_vocab]).reshape(1,len(meds_vocab))
        
 
        # feats.append(np.concatenate([feat_icds, feat_meds, feat_notes], axis = 1))
        result = np.concatenate([feat_icds, feat_meds, feat_notes], axis = 1)
        return result
 
# Run Ray
feats = ray.get([loop.remote(i, df_notes, df_icds_MGH, df_icds_BIDMC, df_meds_MGH, df_meds_BIDMC, notes_vocab_keywords, icds_vocab, meds_vocab) for i in range(length)])

#ray shutdown
ray.shutdown()

# this is your final feature matrix (X)
feat_mat = np.array(feats)


feat_mat2 = np.reshape(feat_mat, (length, 303)) # 303 val is equal to union of length of ICD, Med, and Keywords for CHF


col_names = icds_vocab + meds_vocab + notes_vocab_keywords
df = pd.DataFrame(feat_mat2, columns=col_names)

df.to_csv('FM_2800.csv')