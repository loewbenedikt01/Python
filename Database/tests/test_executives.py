"""
Readable names, normalized roles and entity detection for Form 3/4/5 owners. No network.
"""

from pipeline.executives import is_entity, readable_name, role_of


def test_readable_name():
    assert readable_name('COOK TIMOTHY D') == 'Timothy D. Cook'
    assert readable_name('Khan Sabih') == 'Sabih Khan'
    assert readable_name("O'BRIEN DEIRDRE") == "Deirdre O'Brien"
    assert readable_name('MCDONALD ROBERT A') == 'Robert A. McDonald'
    assert readable_name('SMITH JOHN R JR') == 'John R. Smith Jr.'
    assert readable_name('VAN DER BERG JOHN A') == 'John A. van der Berg'


def test_role():
    assert role_of('Chief Executive Officer', True, True) == 'CEO'
    assert role_of('Chairman and CEO', True, True) == 'CEO'
    assert role_of('Senior Vice President, CFO', True, False) == 'CFO'
    assert role_of('SVP, GC and Secretary', True, False) == 'General Counsel'
    assert role_of('Executive Chair', True, True) == 'Chair'
    assert role_of('Vice President', True, False) == 'Other'          # 'Vice President' is not President
    assert role_of('President and COO', True, False) == 'COO'
    assert role_of(None, False, True) == 'Director'


def test_business_line_heads_are_not_company_ceo():
    assert role_of('CEO CCB', True, False) == 'Other'
    assert role_of('Co-CEO CIB', True, False) == 'Other'
    assert role_of('CEO Asset & Wealth Management', True, False) == 'Other'
    assert role_of('President, Global Banking', True, False) == 'Other'
    assert role_of('President and Chief Executive Officer', True, True) == 'CEO'
    assert role_of('CEO and Director', True, True) == 'CEO'
    assert role_of('Executive Vice President & CFO', True, False) == 'CFO'


def test_entities_are_not_people():
    assert is_entity('BERKSHIRE HATHAWAY INC') and is_entity('Smith Family Trust') and is_entity('KKR Capital Partners LP')
    assert not is_entity('COOK TIMOTHY D') and not is_entity('DIMON JAMES')
