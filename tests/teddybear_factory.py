import numpy as np


def generate_teddybears(production_plan: dict, fluffiness: dict, sigma: float = 0.01) -> list:
    """Creates a list of pairs (plushie, [fluffiness]).

    An information channel can be treated as a device changing non-deterministically labels X (called plushies) into
    measured value Y (called fluffiness). Mutual information measures how much information about plushies we
    can retrieve just by observing fluffiness.
    For each plushie we generate fluffiness randomly around a given target value.

    Parameters
    ----------
    production_plan : dict
        dictionary of plushies (str or int) -> amount created (int)

    fluffiness : dict (label, float)
        dictionary of plushies (str or int) -> target fluffiness (float)

    sigma : float
        positive parameter resulting non-deterministic error to fluffiness around the mean value

    Returns
    -------
    list
        list of pairs describing created plushies. Example:
        [
            ('black_teddy', [0.001]),
            ('black_teddy', [-0.001]),
            ('white_teddy', [-0.002]),
            ...
        ]
    """
    delivery = []
    for plushie, quantity in production_plan.items():
        for _ in range(quantity):
            outcome_fluffiness = (
                fluffiness[plushie] +
                sigma * np.random.normal()
            )
            delivery.append(
                (plushie, [outcome_fluffiness])        
            )
    return delivery


def example():  # pragma: no cover
    """In this example we create a list with 30 plushies - 10 of each kind. There are two colors of teddybears,
    both have the same fluffiness, so some information is lost. We expect that teddybears can be distinguished
    from a bunny, unless `sigma` is very high. In such case fluffiness would not give any information."""
    print("Extra cute example")
    print("******************")
    print("\n\nProduction")
    prod_plan = {
        "black_teddy": 10,
        "white_teddy": 10,
        "giant_bunny": 10,
    }
    for k, v in prod_plan.items():
        print(k, ":", v)

    print("\n\nFluffiness")
    fluff = {
        "black_teddy": 0.,
        "white_teddy": 0.,
        "giant_bunny": 1.,
    }
    for k, v in fluff.items():
        print(k, ":", v)

    print("\n\nEffect")
    for u in generate_teddybears(prod_plan, fluff, sigma=1e-4):
        print(u)
