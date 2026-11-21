"""Classify an English message locally. This does not send an SMS."""
from pathlib import Path
import argparse
import joblib


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--text',required=True,help='English SMS text')
    args=parser.parse_args()
    if not args.text.strip():
        parser.error('Text cannot be blank.')
    path=Path(__file__).parent/'outputs'/'model.joblib'
    if not path.exists():
        parser.error('Run python main.py first.')
    # Load only your own trusted generated model files, never arbitrary joblib files.
    pipeline=joblib.load(path)
    label=int(pipeline.predict([args.text])[0])
    score=float(pipeline.predict_proba([args.text])[0,1])
    print(f'label={"spam" if label else "ham"}; spam_score={score:.4f} (uncalibrated)')


if __name__=='__main__':
    main()
