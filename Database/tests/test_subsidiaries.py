"""
Exhibit 21 parsing: HTML tables and the usual text layouts. No network.
"""

from pipeline.subsidiaries import is_significant_only, parse_ex21

TABLE = """
<p>Exhibit 21.1</p><p>Subsidiaries of Apple Inc.*</p>
<table>
<tr><td><b>Name</b></td><td></td><td><b>Jurisdiction of Incorporation</b></td></tr>
<tr><td>Apple Asia Limited</td><td>&#160;</td><td>Hong Kong</td></tr>
<tr><td>Apple Asia LLC</td><td></td><td>Delaware, U.S.</td></tr>
<tr><td>Apple Operations Mexico, S.A. de C.V.</td><td>Mexico</td><td>100%</td></tr>
</table>
<p>* Pursuant to Item 601(b)(21)(ii) of Regulation S-K, the names of other subsidiaries ... are omitted.</p>
"""

TEXT = """<pre>
SUBSIDIARIES OF EXAMPLE CORP
Name of Subsidiary                          Jurisdiction
Example Holdings LLC ....................... Delaware
Example Europe GmbH                          Germany
Example Japan K.K. (Japan)
Example Finance Company, a Nevada corporation
Example Services Ltd.
</pre>"""


def test_html_table():
    subs = parse_ex21(TABLE)
    assert subs == [{'name': 'Apple Asia Limited', 'jurisdiction': 'Hong Kong'},
                    {'name': 'Apple Asia LLC', 'jurisdiction': 'Delaware, U.S.'},
                    {'name': 'Apple Operations Mexico, S.A. de C.V.', 'jurisdiction': 'Mexico'}]


def test_text_layouts():
    subs = {s['name']: s['jurisdiction'] for s in parse_ex21(TEXT)}
    assert subs == {'Example Holdings LLC': 'Delaware', 'Example Europe GmbH': 'Germany',
                    'Example Japan K.K.': 'Japan', 'Example Finance Company': 'Nevada',
                    'Example Services Ltd.': ''}


def test_header_row_with_date_and_jurisdiction_heading_is_skipped():
    doc = """<table><tr><td>December 31, 2025 Name</td><td>Organized Under The Laws Of</td></tr>
             <tr><td>JPMorgan Chase Bank, National Association</td><td>United States</td></tr>
             <tr><td>J.P. Morgan SE</td><td>Germany</td></tr></table>"""
    assert [s['name'] for s in parse_ex21(doc)] == ['JPMorgan Chase Bank, National Association', 'J.P. Morgan SE']


def test_numbered_rows_and_single_row_table():
    doc = """<table><tr><td>Subsidiary</td><td>State of Incorporation</td></tr>
             <tr><td>1.</td><td>DTE Electric Company</td><td>Michigan</td></tr>
             <tr><td>2</td><td>234DP Aviation, LLC</td><td>Delaware</td></tr></table>"""
    assert parse_ex21(doc) == [{'name': 'DTE Electric Company', 'jurisdiction': 'Michigan'},
                               {'name': '234DP Aviation, LLC', 'jurisdiction': 'Delaware'}]
    one = """<table><tr><td>Name of Corporation</td><td>State of Incorporation</td></tr>
             <tr><td>Union Pacific Railroad Company</td><td>Delaware</td></tr></table>"""
    assert parse_ex21(one) == [{'name': 'Union Pacific Railroad Company', 'jurisdiction': 'Delaware'}]


def test_bullet_list():
    doc = "<div>&#8226;&#160;&#160;&#160;&#160;Paylocity Corporation, an Illinois corporation</div>"           "<div>&#8226;&#160;&#160;&#160;&#160;VidGrid Inc., a Delaware corporation</div>"
    assert parse_ex21(doc) == [{'name': 'Paylocity Corporation', 'jurisdiction': 'Illinois'},
                               {'name': 'VidGrid Inc.', 'jurisdiction': 'Delaware'}]


def test_scope():
    from pipeline.subsidiaries import scope_of
    assert scope_of('<p>Exhibit 21.1 Significant Subsidiaries of the Company. Listed below are the significant '
                    'subsidiaries (as defined in Rule 1-02(w))</p>') == 'significant_only'          # UNH 2025
    assert scope_of("<p>the following is a list of JPMorgan Chase & Co.'s significant legal entity subsidiaries</p>") \
        == 'significant_only'                                                                  # JPM
    assert scope_of('<p>subsidiaries not listed would not, in the aggregate, constitute a "significant subsidiary"</p>') \
        == 'omits_insignificant'                                                               # UNH 2024
    assert scope_of(TABLE) == 'omits_insignificant'                                            # AAPL: 601(b)(21)(ii)
    assert scope_of('<p>Below is a list of all of the Registrant subsidiaries</p>') == 'all'   # CDNS
    assert not is_significant_only(TABLE)
