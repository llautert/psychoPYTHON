import argparse
from models import *
from data_management import *
from predict import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--typemodel", help="select type model")
    parser.add_argument("-d", "--typedeepmodel", help="select type deep model")
    parser.add_argument("-t", "--train", help="train model", action="store_true")
    parser.add_argument("-p", "--predict", help="file to read for prediction", action="store_true")
    parser.add_argument("--prediction", help="type of the prediction")

    parser.add_argument("-l", "--labeled_data", help="there is labeled data", action="store_true")
    parser.add_argument("--text_data", help="there is text data", action="store_true")
    parser.add_argument("-c", "--category_data", help="there is category data", action="store_true")
    parser.add_argument("-f", "--few_important_features", help="there is few important features", action="store_true")
    parser.add_argument("-j", "--just_looking", help="you are just looking the code", action="store_true")

    args = parser.parse_args()

    if args.train:
        if args.typemodel:
            X, y = load_processed_data_test()
            if args.typemodel == 'autoscikit':
                autoscikit(X, y, args.prediction, args.labeled_data, args.text_data, args.category_data, args.few_important_features, args.just_looking)
        if args.typedeepmodel:
            if args.typedeepmodel == '1net':
                deep_1net(X, y)

    elif args.predict:
        X = load_predict_data_test()
        if args.typemodel:
            print(predict(args.typemodel, X))
        if args.typedeepmodel:
            print(predict_deep(args.typedeepmodel, X))
