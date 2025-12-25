import pandas as pd

def create_dataset():
    data = {
        "text": [
            # Physical
            "Regular exercise improves physical health and stamina",
            "Walking daily helps maintain a healthy body",
            "Eating nutritious food keeps the body fit",
            "Physical activity strengthens muscles and bones",
            "Adequate sleep is important for physical recovery",

            # Educational
            "Education improves knowledge and critical thinking",
            "Students learn new skills through education",
            "Reading books enhances learning and understanding",
            "Online courses help students gain education",
            "Teachers play an important role in education",

            # Spiritual
            "Meditation helps achieve inner peace",
            "Prayer strengthens spiritual connection",
            "Spiritual practices calm the mind",
            "Faith brings spiritual strength and hope",
            "Mindfulness improves spiritual awareness"
        ],
        "label": [
            "Physical", "Physical", "Physical", "Physical", "Physical",
            "Educational", "Educational", "Educational", "Educational", "Educational",
            "Spiritual", "Spiritual", "Spiritual", "Spiritual", "Spiritual"
        ]
    }

    df = pd.DataFrame(data)
    return df


