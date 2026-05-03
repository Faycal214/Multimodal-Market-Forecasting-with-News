from src.models.predict import save_submission

def main():
    preds = save_submission("submission.csv")
    print(preds.head())

if __name__ == "__main__":
    main()