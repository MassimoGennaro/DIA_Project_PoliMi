import numpy as np
import matplotlib.pyplot as plt

from learner import *
from modules import *
from utils_functions import *

###### DATI DEL PROBLEMA #####

###### Dati delle Persone

# considero le categorie sempre come numeri interi
categories = {0: ("a", "f"), 1: ("y", "f"), 2: ("y", "u")}

# le features servono sono per capire a cosa corrispondono i valori
features = {"Age": ("y", "a"), "Familiarity": ("f", "u")}

# ogni tupla è in ordine secondo le variabili: (e.g. (prima var, seconda var, ecc))
feature_space = [("y", "f"), ("y", "u"), ("a", "f"), ("a", "u")]

# ogni categoria ha una probabilità a priori
# sono equiprobabili
prior_categories = [1, 1, 1]

# l'environment è suddiviso in fasi, in cui in ciascuna cambiano le probabilità


# probabilità delle 3 classi per ogni candidato.
# deve essere ampliata per considerare le fasi: lista di liste di liste: [phase][category][arm-chance]
p_categories = np.array([[[0.85, 0.55, 0.50, 0.25, 0.15],
                          [0.95, 0.90, 0.60, 0.05, 0.00],
                          [0.90, 0.45, 0.20, 0.15, 0.05]]])

###### Dati dei Candidati ######
# abbiamo 5 candidati di prezzzi diversi
n_arms = 5
# valori dei candidati
arms_candidates = np.array([5, 10, 15, 20, 25])

# per ogni categoria calcolo il valore atteso di ogni arm
exp_values_categories = np.multiply(p_categories[0], arms_candidates)

# per ogni categoria trovo il valore atteso del best arm
best_exp_value_categories = np.max(exp_values_categories, axis=1).T

# Scelgo best arm per category, el0 = best expected value per category 0
# [0.15, 0.1, 0.1, 0.35, 0.35]                          [0.75 1.0  1.5  7.0   8.75]
# [0.15, 0.1, 0.1, 0.35, 0.35] * [5, 10, 15, 20, 25] =  [0.75 1.0  1.5  7.0   8.75] -(max)> [8.75, 8.75, 8.75]
# [0.15, 0.1, 0.1, 0.35, 0.35]                          [0.75 1.0  1.5  7.0   8.75]
opt_arm_categories = np.argmax(exp_values_categories, axis=1).T

### EXPERIMENT PARAMETERS ###

# T è il numero di persone
T = 400

# numero di fasi in cui cambiano le probabilità delle categorie
num_phases = len(p_categories)
phases = create_phases(T, num_phases)

print("phases:{}".format(phases))

# numero di esperimenti
n_experiments = 10

experiments_logs = []

print("Informazioni\n")
print("A = \n", p_categories)
print("B = \n", arms_candidates)
print("A*B = \n", exp_values_categories)
print("max(A*B, axis=1) = \n", best_exp_value_categories)
print("argmax(A*B, axis=1) = \n", opt_arm_categories)
print("Horizon = ", T, "\nphases = \n", phases, "\nn_experiments = \n", n_experiments)

###### RUN EXPERIMENTS ######


experiments_logs = []
beta_parameters_list = []

for e in range(n_experiments):
    print("experiment: {}".format(e))
    # ad ogni experiment creiamo un nuovo Environment
    # utilizziamo un environment adatto, rende la reward in base a category e candidato
    environment = Personalized_Environment(arms_candidates, p_categories, phases, T)

    # utilizziamo un person_manager per gestire la creazione di nuove persone
    p_manager = Person_Manager(categories, p_categories[0], features, T)
    # utilizziamo un context_manager per gestire la gestione dei contesti e learner
    c_manager = Context_Manager(n_arms, feature_space, arms_candidates)
    # general gestisce la logica
    general = General(p_manager, c_manager, environment)

    # general itera per t round, restituendo le statistiche finali dell'esperimento
    experiment_log = general.play_experiment(T)

    # memorizzo per ogni esperimento i beta parameters finali
    beta_parameters_list.append(general.context_manager.contexts_set[0].learner.beta_parameters)

    # appendo le statistiche finali
    experiments_logs.append(experiment_log)

print(len(experiments_logs))
###### SHOW REGRET ######


# TODO gli expected values dipendono dalla fase
total_regret_list = []
# per ogni persona con una categoria, regret_t è valore atteso migliore della categoria
# meno valore atteso dell'arm scelto per tale categoria
for e in range(n_experiments):
    print("experiment: {}".format(e))
    regret_list = []
    regret_t = 0
    for reward_t in experiments_logs[e]:
        category_t = reward_t[0]
        arm_chosen_t = reward_t[1]
        best_exp_value = best_exp_value_categories[category_t]
        arm_exp_value = exp_values_categories[category_t][arm_chosen_t]
        regret_t = best_exp_value - arm_exp_value
        regret_list.append(regret_t)
    else:
        total_regret_list.append(regret_list)
else:
    # faccio la media del regret_t di ogni experiment
    average_regret_list = np.cumsum(np.mean(total_regret_list, axis=0))
    print("average_regret_list lenght: {}".format(len(average_regret_list)))
    print("cumulative average_regret_list lenght: {}\n".format(average_regret_list))

# plt.figure(0)
# plt.ylabel("Cumulative Regret in t")
# plt.xlabel("t")
# plt.plot(average_regret_list, 'r')
# plt.legend(["TS generico"])
# plt.show()

#  QUI SI POSSONO OSSERVARE GLI ARMI PIù SCELTI IN TUTTI GLI EXPERIMENT

beta_parameters_list = np.array(beta_parameters_list)
sum_beta_parameters_list = np.sum(beta_parameters_list, axis=0)
print("parametri beta in tutti gli experiment degli arm:\n{}\n".format(sum_beta_parameters_list))

estimated_probs_arms = [x[0] / (x[0] + x[1]) for x in sum_beta_parameters_list]
print("Stima delle probabilità degli arm;\n{}\n".format(estimated_probs_arms))

arms_num_played = np.sum(sum_beta_parameters_list, axis=1)
tot_played = np.sum(arms_num_played)
print("numero arms giocati:\n{}\n\ntotale giocati:\n{}\n"
      .format(arms_num_played, tot_played))
print("valori attesi degli arm in media per le categorie \n{}\n".format(np.mean(exp_values_categories, axis=0)))

print("CORRECTLY ENDED")