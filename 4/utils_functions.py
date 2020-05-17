#utils_functions

def create_phases(T, num_phases):
	# T è il numero di persone
	#print("T:{}".format(T))
	# numero di fasi in cui cambiano le probabilità delle categorie
	#print("num_phases:{}".format(num_phases))
	phase_lenght = T//num_phases
	#print("phase_lenght:{}".format(phase_lenght))
	limit_phases = [0] + [(phase_lenght)*(i+1) for i in range(num_phases)]
	if limit_phases[-1]!=T:
		limit_phases[-1] = T
	#print("limit_phases:{}".format(limit_phases))
	phases = [(i+1, i+phase_lenght) for i in limit_phases[:-1]]
	if phases[-1][1]!=T:
		phases[-1] = (phases[-1][0],T)
	#print("phases:{}".format(phases))
	return phases


#Questa funzione calcola l'expected vaulue NON la best (a meno che non si passi la best_arm)
def best_expected_value(beta_parameters, best_arm, candidates_values):
	#calcolo la sua probabilità di successo
    best_arm_successes = self.learner.beta_parameters[best_arm][0]
    best_arm_failures = self.learner.beta_parameters[best_arm][1]
    best_arm_prob_success = best_arm_successes/(best_arm_successes + best_arm_failures)
    # calcolo il suo valore atteso, deve essere usato per la split condition
    best_expected_value = best_arm_prob_success * candidates_values[best_arm]
    return best_expected_value
