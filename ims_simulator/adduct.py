from cpyMSpec import IsotopePattern
from re import match

def signedAdduct(adduct):
    return adduct if adduct[0] in ('+', '-') else '+' + adduct

def isValidAdduct(adduct):
    if adduct[0] == '-':
        adduct = adduct[1:]
    try:
        IsotopePattern(adduct)
        return True
    except:
        return False

CHARGES = {
    '+H': +1,
    '+K': +1,
    '+Na': +1,
    '-H': -1,
    '+Cl': -1  # this is not a bug, ask Andrew Palmer if you're confused
}

def adductCharge(adduct):
    return CHARGES[adduct]

def splitSumFormula(sf_a):
    return match("(\w+)([+-]\w+)", sf_a).groups()
