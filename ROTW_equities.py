import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

ticker_rotw = [
    # --- UNITED KINGDOM (London Stock Exchange: .L) ---
    'SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'BP.L', 'GSK.L', 'RIO.L', 'REL.L', 'BATS.L', 'DGE.L',
    'LLOY.L', 'BARC.L', 'PRU.L', 'EXPN.L', 'GLEN.L', 'NG.L', 'AAL.L', 'VOD.L', 'BKT.L', 'WCPP.L',
    
    # --- SWITZERLAND (SIX Swiss Exchange: .SW) ---
    'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'CFR.SW', 'ZURN.SW', 'ABBN.SW', 'SIKA.SW', 'LONN.SW', 'ALC.SW',
    'GIVN.SW', 'HOLN.SW', 'SRENH.SW', 'GEBN.SW', 'SCMN.SW', 'BAER.SW', 'LOGN.SW', 'SOON.SW', 'VATN.SW', 'SGSN.SW',

    # --- CANADA (Toronto Stock Exchange: .TO) ---
    'RY.TO', 'TD.TO', 'SHOP.TO', 'ENB.TO', 'CNR.TO', 'CP.TO', 'CNQ.TO', 'BMO.TO', 'ATD.TO', 'TRI.TO',
    'BAM.TO', 'SU.TO', 'MFC.TO', 'BNS.TO', 'ABX.TO', 'WPM.TO', 'IMO.TO', 'SLF.TO', 'CVE.TO', 'TRP.TO',
    'FNV.TO', 'AEM.TO', 'GIB-A.TO', 'TEL.TO', 'BCE.TO', 'RCI-B.TO', 'QLSR.TO', 'WCN.TO', 'H.TO', 'POW.TO',

    # --- AUSTRALIA (Australian Securities Exchange: .AX) ---
    'BHP.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 'MQG.AX', 'WDS.AX', 'WES.AX', 'WOW.AX',
    'FMG.AX', 'RIO.AX', 'GMG.AX', 'TLS.AX', 'APA.AX', 'ALL.AX', 'QAN.AX', 'SUN.AX', 'IAG.AX', 'XRO.AX',

    # --- LATIN AMERICA (Brazil: .SA & Mexico: .MX) ---
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'ABEV3.SA', 'WEGE3.SA', 'ITSA4.SA', 'B3SA3.SA', 'RENT3.SA',
    'AMXL.MX', 'WALMEX.MX', 'FEMSAUBD.MX', 'GMEXICOB.MX', 'GFNORTEO.MX', 'CEMEXCPO.MX', 'BIMBOA.MX', 'ALPEKA.MX', 'GRUMAB.MX', 'ORBIA.MX'
]