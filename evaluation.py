from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

gs_clf = 0
categories = ["INSULT", "ABUSE", "PROFANITY", "OTHER"]


# Use to search for proper model parameters
# Parameters:
# text_clf_svm: Pipeline for Model
# data_frame: Our dataframe with features
# training_label: Our training label list
def search_for_parameters(text_clf_svm, data_frame, training_label, parameters):
    global gs_clf
    print("Starting GridSearchCV...")
    start = datetime.now()
    gs_clf = GridSearchCV(text_clf_svm, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(data_frame, training_label)
    print("Best Score: " + str(gs_clf.best_score_))
    print("Best Params: \n")
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    end = datetime.now()
    duration = end - start
    print("GridSearchCV duration: " + str(duration))
    print(
        "The best parameters are %s with a score of %0.2f"
        % (gs_clf.best_params_, gs_clf.best_score_)
    )


def get_plot(gamma_range, C_range):
    scores = gs_clf.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot
    )
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title("Accuracy")
    plt.show()


# Check how accurate our model was, which categories get switched most
def evaluateModelAccuracy(predicted, original, mapping):
    insult = {
        'INSULT': 0,
        'ABUSE':0,
        'PROFANITY':0,
        'OTHER':0
    }
    abuse = {
        'INSULT': 0,
        'ABUSE': 0,
        'PROFANITY': 0,
        'OTHER': 0
    }
    profanity = {
        'INSULT': 0,
        'ABUSE': 0,
        'PROFANITY': 0,
        'OTHER': 0
    }
    other = {
        'INSULT': 0,
        'ABUSE': 0,
        'PROFANITY': 0,
        'OTHER': 0
    }

    for i in range(len(predicted)):
        if mapping[original[i]] == "INSULT":
            insult[mapping[predicted[i]]] += 1
        elif mapping[original[i]] == "ABUSE":
            abuse[mapping[predicted[i]]] += 1
        elif mapping[original[i]] == "PROFANITY":
            profanity[mapping[predicted[i]]] += 1
        else:
            other[mapping[predicted[i]]] += 1

    # Absolute Plot
    print(insult)
    print(abuse)
    print(profanity)
    print(other)
    fig, ax = plt.subplots()
    insult_list = np.array([insult["INSULT"], abuse["INSULT"], profanity["INSULT"], other["INSULT"]])
    abuse_list = np.array([insult["ABUSE"], abuse["ABUSE"], profanity["ABUSE"], other["ABUSE"]])
    profanity_list = np.array([insult["PROFANITY"], abuse["PROFANITY"], profanity["PROFANITY"], other["PROFANITY"]])
    other_list = np.array([insult["OTHER"], abuse["OTHER"], profanity["OTHER"], other["OTHER"]])

    bottom_list = np.array([0.0,0.0,0.0,0.0])

    ax.bar(categories, insult_list, 0.35, label="INSULT")
    bottom_list = np.add(bottom_list, insult_list)
    ax.bar(categories, abuse_list, 0.35, bottom=bottom_list, label="ABUSE")
    bottom_list = np.add(bottom_list, abuse_list)
    ax.bar(categories, profanity_list, 0.35, bottom=bottom_list, label="PROFANITY")
    bottom_list = np.add(bottom_list, profanity_list)
    ax.bar(categories, other_list, 0.35, bottom=bottom_list, label="OTHER")

    plt.legend()
    ax.set_title("Kategorische Übersicht (Absolute Werte)")
    ax.set_xlabel("Kategorien")
    ax.set_ylabel("Klassifikation")
    plt.show()

    # Relative Plot
    insult = convert_abs_to_rel(insult)
    abuse = convert_abs_to_rel(abuse)
    profanity = convert_abs_to_rel(profanity)
    other = convert_abs_to_rel(other)
    fig, ax = plt.subplots()
    insult_list = np.array([insult["INSULT"], abuse["INSULT"], profanity["INSULT"], other["INSULT"]])
    abuse_list = np.array([insult["ABUSE"], abuse["ABUSE"], profanity["ABUSE"], other["ABUSE"]])
    profanity_list = np.array([insult["PROFANITY"], abuse["PROFANITY"], profanity["PROFANITY"], other["PROFANITY"]])
    other_list = np.array([insult["OTHER"], abuse["OTHER"], profanity["OTHER"], other["OTHER"]])

    bottom_list = np.array([0.0, 0.0, 0.0, 0.0])

    ax.bar(categories, insult_list, 0.35, label="INSULT")
    bottom_list = np.add(bottom_list, insult_list)
    ax.bar(categories, abuse_list, 0.35, bottom=bottom_list, label="ABUSE")
    bottom_list = np.add(bottom_list, abuse_list)
    ax.bar(categories, profanity_list, 0.35, bottom=bottom_list, label="PROFANITY")
    bottom_list = np.add(bottom_list, profanity_list)
    ax.bar(categories, other_list, 0.35, bottom=bottom_list, label="OTHER")

    plt.legend()
    ax.set_title("Kategorische Übersicht (Relative Werte)")
    ax.set_xlabel("Kategorien")
    ax.set_ylabel("Klassifikation")
    plt.show()


def convert_abs_to_rel(dict):
    relative_frame = {
        'INSULT': 0.0,
        'ABUSE': 0.0,
        'PROFANITY': 0.0,
        'OTHER': 0.0
    }
    abs = 0
    for category in categories:
        abs += dict[category]
    for category in categories:
        relative_frame[category] = dict[category] / abs
    return relative_frame
