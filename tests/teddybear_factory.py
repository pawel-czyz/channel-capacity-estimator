import numpy as np


def generate_teddybears(production_plan, fluffiness, precision=0.01):
    """Creates a list of pairs (plushie, [fluffiness])

    Parameters
    ----------
    production_plan : dictionary (label, int)
        dictionary of plushies -> amount created

    fluffiness : dictionary (label, float)
        dictionary of plushies -> target fluffiness

    precision : float
        precision of fluffiness production

    Returns
    -------
    list
        list of pairs describing created plushies. Example:
        [
            ('black_teddy',  0.001),
            ('black_teddy', -0.001),
            ('white_teddy', -0.002),
            ...
        ]
    """
    delivery = []
    for plushie, quantity in production_plan.items():
        for _ in range(quantity):
            outcome_fluffiness = (
                    fluffiness[plushie] +
                    precision * np.random.normal()
            )
            delivery.append(
                (plushie, [outcome_fluffiness])        
            )
    return delivery


if __name__ == "__main__":
    print("Extra cute example")
    print("******************")
    print("\n\nProduction")
    prod_plan = {
        "black_teddy": 10,
        "white_teddy": 10,
        "giant_bunny": 10
    }
    for k, v in prod_plan.items():
        print(k, ":", v)

    print("\n\nFluffiness")
    fluff = {
        "black_teddy": 0.,
        "white_teddy": 0.,
        "giant_bunny": 1.
    }
    for k, v in fluff.items():
        print(k, ":", v)

    print("\n\nEffect")
    for u in generate_teddybears(prod_plan, fluff, 1e-4):
        print(u)
