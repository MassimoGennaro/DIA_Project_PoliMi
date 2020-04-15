import numpy as np

click_functions = {
    "Young-Familiar": {
        "Morning": lambda x: (1 - np.exp(-x)) * 100,
        "Evening": lambda x: (1 - np.exp(-x)) * 150,
        "Weekend": lambda x: (1 - np.exp(-x)) * 300
    },
    "Adult-Familiar": {
        "Morning": lambda x: (1 - np.exp(-x)) * 10,
        "Evening": lambda x: (1 - np.exp(-x)) * 100,
        "Weekend": lambda x: (1 - np.exp(-x)) * 150
    },
    "Young-NotFamiliar": {
        "Morning": lambda x: (1 - np.exp(-x)) * 70,
        "Evening": lambda x: (1 - np.exp(-x)) * 50,
        "Weekend": lambda x: (1 - np.exp(-x)) * 100
    },
}
