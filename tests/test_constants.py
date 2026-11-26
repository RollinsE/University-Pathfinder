from src.constants import BUILDINGS, CLASS_NAMES, NUM_CLASSES, canonical_index, normalise_name


def test_eight_buildings_in_alphabetical_order():
    assert NUM_CLASSES == 8
    assert CLASS_NAMES == sorted(CLASS_NAMES)
    assert len({b["slug"] for b in BUILDINGS}) == NUM_CLASSES


def test_every_spelling_maps_to_one_class():
    library = CLASS_NAMES.index("Sir Michael Cobham Library")
    for name in (
        "Sir-Michael-Cobham-Library",
        "The_Sir_Michael_Cobham_Library",
        "Sir Micheal Cobham Library",
        "sir_michael_cobham_library",
    ):
        assert canonical_index(name) == library, name

    assert canonical_index("Dorset") == CLASS_NAMES.index("Dorset House")
    assert canonical_index("Poole-Gateway-Building") == CLASS_NAMES.index("Poole Gateway Building")
    assert canonical_index("Poole_House") == CLASS_NAMES.index("Poole House")


def test_poole_house_and_poole_gateway_stay_apart():
    assert canonical_index("Poole House") != canonical_index("Poole Gateway")


def test_unknown_names_are_rejected():
    assert canonical_index("Bicycle") is None
    assert canonical_index("buildings") is None


def test_normalise_name():
    assert normalise_name("  The Sir-Micheal_Cobham  Library ") == "sir michael cobham library"
