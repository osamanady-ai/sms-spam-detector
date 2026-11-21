"""English SMS spam baseline with canonical deduplication before splitting."""
from resources import ensure_assets
ensure_assets()

from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'


def canonical(text):
    """Group casing/whitespace/number variants to reduce duplicate leakage."""
    return re.sub(r'\s+', ' ', re.sub(r'\d+', ' NUM ', str(text).lower())).strip()


def load_clean():
    raw=pd.read_csv(BASE/'data'/'SMSSpamCollection',sep='\t',header=None,names=['label','text'],quoting=3)
    assert set(raw.label.unique())=={'ham','spam'}
    frame=raw.dropna().copy()
    frame['key']=frame.text.map(canonical)
    frame=frame.loc[frame.key.str.len()>0]
    # Remove ambiguous canonical groups instead of arbitrarily choosing a label.
    ambiguous=frame.groupby('key').label.nunique()
    conflicts=set(ambiguous[ambiguous>1].index)
    frame=frame.loc[~frame.key.isin(conflicts)].drop_duplicates('key').reset_index(drop=True)
    frame['target']=frame.label.eq('spam').astype(int)
    return raw,frame,len(conflicts)


def main():
    OUT.mkdir(exist_ok=True)
    raw,frame,conflicts=load_clean()
    train,test=train_test_split(frame,test_size=.2,random_state=42,stratify=frame.target)
    assert not set(train.key)&set(test.key)
    assert not set(train.index)&set(test.index)
    pipe=Pipeline([('tfidf',TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,sublinear_tf=True)),
                   ('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))])
    search=GridSearchCV(pipe,{'model__C':[.5,1.,4.]},cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='f1',n_jobs=1)
    # Vocabulary and IDF are learned inside each CV training fold, not on test text.
    search.fit(train.text,train.target)
    pred=search.predict(test.text)
    probability=search.predict_proba(test.text)[:,1]
    baseline=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),train.target).predict(np.zeros((len(test),1)))
    scores={'raw_rows':len(raw),'clean_unique_rows':len(frame),'removed_rows':len(raw)-len(frame),
            'conflicting_canonical_groups_removed':conflicts,'train_rows':len(train),'test_rows':len(test),
            'test_spam_rows':int(test.target.sum()),'best_C':search.best_params_['model__C'],
            'train_cv_spam_f1':float(search.best_score_),'test_accuracy':float(accuracy_score(test.target,pred)),
            'spam_precision':float(precision_score(test.target,pred,zero_division=0)),
            'spam_recall':float(recall_score(test.target,pred,zero_division=0)),
            'spam_f1':float(f1_score(test.target,pred,zero_division=0)),
            'baseline_accuracy':float(accuracy_score(test.target,baseline)),
            'baseline_spam_f1':float(f1_score(test.target,baseline,zero_division=0))}
    (OUT/'metrics.json').write_text(json.dumps(scores,indent=2))
    (OUT/'classification_report.txt').write_text(classification_report(test.target,pred,target_names=['ham','spam'],zero_division=0))
    # Store IDs and outcomes, not example phone numbers or clickable spam URLs.
    predictions=pd.DataFrame({'row_id':test.index,'actual':test.target.to_numpy(),'predicted':pred,'spam_score':probability})
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    predictions.loc[predictions.actual != predictions.predicted].to_csv(OUT/'error_analysis.csv',index=False)
    pd.DataFrame({'row_id':train.index,'split':'train'}).to_csv(OUT/'train_rows.csv',index=False)
    joblib.dump(search.best_estimator_,OUT/'model.joblib')
    tfidf=search.best_estimator_.named_steps['tfidf']
    weights=search.best_estimator_.named_steps['model'].coef_[0]
    terms=pd.DataFrame({'term':tfidf.get_feature_names_out(),'weight':weights})
    pd.concat([terms.nsmallest(15,'weight'),terms.nlargest(15,'weight')]).to_csv(OUT/'indicative_terms.csv',index=False)
    pd.DataFrame(search.cv_results_)[['param_model__C','mean_test_score','std_test_score','rank_test_score']].to_csv(OUT/'cv_results.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    ConfusionMatrixDisplay.from_predictions(test.target,pred,display_labels=['ham','spam'],ax=axes[0],colorbar=False,cmap='Blues')
    axes[0].set_title('Held-out confusion matrix')
    PrecisionRecallDisplay.from_predictions(test.target,probability,ax=axes[1],name='TF-IDF + Logistic Regression')
    axes[1].set_title('Spam precision-recall tradeoff')
    fig.tight_layout();fig.savefig(OUT/'evaluation.png',dpi=150);plt.close(fig)
    (OUT/'report.md').write_text(f'''# English SMS spam results

- {len(raw)} raw lines; {len(frame)} canonical unique messages after cleaning.
- {len(train)} training messages; {len(test)} test messages, including {int(test.target.sum())} spam.
- Test spam precision: {scores['spam_precision']:.3f}; recall: {scores['spam_recall']:.3f}; F1: {scores['spam_f1']:.3f}.
- Test accuracy: {scores['test_accuracy']:.3f}; majority baseline accuracy: {scores['baseline_accuracy']:.3f}.
- Majority baseline spam F1: {scores['baseline_spam_f1']:.3f}; 5-fold training CV spam F1: {scores['train_cv_spam_f1']:.3f}.

Majority accuracy hides the minority-spam problem, so precision, recall and F1 matter.
Canonical keys lowercase text, normalize whitespace and replace digit sequences; each key
is kept once before splitting, and conflicting-label keys are discarded. No key appears
in both splits. Semantically similar templates may still remain; this is not a temporal,
sender-held-out or real deployment benchmark. Vocabulary and IDF fit only on training folds.
Scores are not calibrated probabilities. Decision threshold is the model default (0.5);
the precision-recall plot is diagnostic, not a threshold tuned against the test set.
Historical English SMS is different from Egyptian Arabic, WhatsApp and current scams.
Do not auto-delete messages. Review false positives and never follow links in raw spam data.

![Evaluation](evaluation.png)
''')
    print(json.dumps(scores,indent=2))


if __name__=='__main__':
    main()
