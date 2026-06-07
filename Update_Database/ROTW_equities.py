import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

ticker_rotw = {
    # --- UNITED KINGDOM (London Stock Exchange: .L) ---
    'SHEL.L': 'Shell plc', 'AZN.L': 'AstraZeneca plc', 'HSBA.L': 'HSBC Holdings plc',
    'ULVR.L': 'Unilever plc', 'BP.L': 'BP plc', 'GSK.L': 'GSK plc',
    'RIO.L': 'Rio Tinto plc', 'REL.L': 'RELX plc', 'BATS.L': 'British American Tobacco p.l.c.',
    'DGE.L': 'Diageo plc', 'LLOY.L': 'Lloyds Banking Group plc', 'BARC.L': 'Barclays plc',
    'PRU.L': 'Prudential plc', 'EXPN.L': 'Experian plc', 'GLEN.L': 'Glencore plc',
    'NG.L': 'National Grid plc', 'AAL.L': 'Anglo American plc', 'VOD.L': 'Vodafone Group plc',
    'BKT.L': 'Babcock International Group PLC', 'WCPP.L': 'Whitbread PLC',

    # --- SWITZERLAND (SIX Swiss Exchange: .SW) ---
    'NESN.SW': 'Nestlé S.A.', 'ROG.SW': 'Roche Holding AG', 'NOVN.SW': 'Novartis AG',
    'UBSG.SW': 'UBS Group AG', 'CFR.SW': 'Compagnie Financière Richemont S.A.',
    'ZURN.SW': 'Zurich Insurance Group AG', 'ABBN.SW': 'ABB Ltd', 'SIKA.SW': 'Sika AG',
    'LONN.SW': 'Lonza Group AG', 'ALC.SW': 'Alcon Inc.', 'GIVN.SW': 'Givaudan SA',
    'HOLN.SW': 'Holcim Ltd', 'SRENH.SW': 'Swiss Re AG', 'GEBN.SW': 'Geberit AG',
    'SCMN.SW': 'Swisscom AG', 'BAER.SW': 'Julius Bär Gruppe AG', 'LOGN.SW': 'Logitech International S.A.',
    'SOON.SW': 'Sonova Holding AG', 'VATN.SW': 'VAT Group AG', 'SGSN.SW': 'SGS S.A.',

    # --- CANADA (Toronto Stock Exchange: .TO) ---
    'RY.TO': 'Royal Bank of Canada', 'TD.TO': 'The Toronto-Dominion Bank',
    'SHOP.TO': 'Shopify Inc.', 'ENB.TO': 'Enbridge Inc.', 'CNR.TO': 'Canadian National Railway Company',
    'CP.TO': 'Canadian Pacific Kansas City Limited', 'CNQ.TO': 'Canadian Natural Resources Limited',
    'BMO.TO': 'Bank of Montreal', 'ATD.TO': 'Alimentation Couche-Tard Inc.',
    'TRI.TO': 'Thomson Reuters Corporation', 'BAM.TO': 'Brookfield Asset Management Ltd.',
    'SU.TO': 'Suncor Energy Inc.', 'MFC.TO': 'Manulife Financial Corporation',
    'BNS.TO': 'Bank of Nova Scotia', 'ABX.TO': 'Barrick Gold Corporation',
    'WPM.TO': 'Wheaton Precious Metals Corp.', 'IMO.TO': 'Imperial Oil Limited',
    'SLF.TO': 'Sun Life Financial Inc.', 'CVE.TO': 'Cenovus Energy Inc.',
    'TRP.TO': 'TC Energy Corporation', 'FNV.TO': 'Franco-Nevada Corporation',
    'AEM.TO': 'Agnico Eagle Mines Limited', 'GIB-A.TO': 'CGI Inc.',
    'TEL.TO': 'TELUS Corporation', 'BCE.TO': 'BCE Inc.', 'RCI-B.TO': 'Rogers Communications Inc.',
    'QLSR.TO': 'Dollarama Inc.', 'WCN.TO': 'Waste Connections, Inc.', 'H.TO': 'Hydro One Limited',
    'POW.TO': 'Power Corporation of Canada',

    # --- AUSTRALIA (Australian Securities Exchange: .AX) ---
    'BHP.AX': 'BHP Group Limited', 'CBA.AX': 'Commonwealth Bank of Australia',
    'CSL.AX': 'CSL Limited', 'NAB.AX': 'National Australia Bank Limited',
    'WBC.AX': 'Westpac Banking Corporation', 'ANZ.AX': 'ANZ Group Holdings Limited',
    'MQG.AX': 'Macquarie Group Limited', 'WDS.AX': 'Woodside Energy Group Ltd',
    'WES.AX': 'Wesfarmers Limited', 'WOW.AX': 'Woolworths Group Limited',
    'FMG.AX': 'Fortescue Ltd', 'RIO.AX': 'Rio Tinto Limited', 'GMG.AX': 'Goodman Group',
    'TLS.AX': 'Telstra Group Limited', 'APA.AX': 'APA Group', 'ALL.AX': 'Aristocrat Leisure Limited',
    'QAN.AX': 'Qantas Airways Limited', 'SUN.AX': 'Suncorp Group Limited',
    'IAG.AX': 'Insurance Australia Group Limited', 'XRO.AX': 'Xero Limited',

    # --- LATIN AMERICA (Brazil: .SA & Mexico: .MX) ---
    'VALE3.SA': 'Vale S.A.', 'PETR4.SA': 'Petróleo Brasileiro S.A.',
    'ITUB4.SA': 'Itaú Unibanco Holding S.A.', 'BBDC4.SA': 'Banco Bradesco S.A.',
    'BBAS3.SA': 'Banco do Brasil S.A.', 'ABEV3.SA': 'Ambev S.A.',
    'WEGE3.SA': 'WEG S.A.', 'ITSA4.SA': 'Itaúsa S.A.', 'B3SA3.SA': 'B3 S.A.',
    'RENT3.SA': 'Localiza Rent a Car S.A.', 'AMXL.MX': 'América Móvil, S.A.B. de C.V.',
    'WALMEX.MX': 'Wal-Mart de México, S.A.B. de C.V.',
    'FEMSAUBD.MX': 'Fomento Económico Mexicano, S.A.B. de C.V.',
    'GMEXICOB.MX': 'Grupo México, S.A.B. de C.V.',
    'GFNORTEO.MX': 'Grupo Financiero Banorte, S.A.B. de C.V.',
    'CEMEXCPO.MX': 'Cemex, S.A.B. de C.V.', 'BIMBOA.MX': 'Grupo Bimbo, S.A.B. de C.V.',
    'ALPEKA.MX': 'Alfa, S.A.B. de C.V.', 'GRUMAB.MX': 'Gruma, S.A.B. de C.V.',
    'ORBIA.MX': 'Orbia Advance Corporation, S.A.B. de C.V.'
}