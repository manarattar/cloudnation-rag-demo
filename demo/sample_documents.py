"""
Sample Dutch tax law documents for the demo.
These represent the four corpus types: legislation, case law, policy, e-learning.
"""

DOCUMENTS = [
    {
        "doc_id": "IB2024-3114",
        "doc_type": "legislation",
        "doc_title": "Wet Inkomstenbelasting 2024",
        "article": "Artikel 3.114",
        "paragraph": "Lid 1-2",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Artikel 3.114 — Persoonsgebonden aftrek. "
            "Lid 1: De persoonsgebonden aftrek wordt in aanmerking genomen bij het bepalen "
            "van het belastbare inkomen uit werk en woning (box 1). "
            "Lid 2: De aftrek bedraagt het bedrag van de in het kalenderjaar op de "
            "belastingplichtige drukkende persoonsgebonden aftrekposten. "
            "Lid 3: Niet-benutte persoonsgebonden aftrek wordt verrekend met inkomen "
            "uit de overige boxen conform artikel 6.1 lid 3."
        ),
    },
    {
        "doc_id": "IB2024-221",
        "doc_type": "legislation",
        "doc_title": "Wet Inkomstenbelasting 2024",
        "article": "Artikel 2.21",
        "paragraph": "Box 1 tarieven",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Artikel 2.21 — Tarieven box 1 (belastbaar inkomen uit werk en woning) 2024. "
            "Schijf 1: inkomen tot € 75.518 — tarief 36,97%. "
            "Schijf 2: inkomen vanaf € 75.518 — tarief 49,50%. "
            "AOW-gerechtigden (geboren voor 1 januari 1946): schijf 1 tarief 19,07%, "
            "schijf 2 tarief 36,97%, schijf 3 tarief 49,50%. "
            "De heffingskortingen worden in mindering gebracht op de verschuldigde belasting."
        ),
    },
    {
        "doc_id": "IB2024-364",
        "doc_type": "legislation",
        "doc_title": "Wet Inkomstenbelasting 2024",
        "article": "Artikel 3.64",
        "paragraph": "Thuiswerkaftrek",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Artikel 3.64 — Thuiswerkkosten. "
            "Werkgevers mogen een onbelaste thuiswerkvergoeding verstrekken van maximaal "
            "€ 2,35 per thuiswerkdag (2024). "
            "Deze vergoeding is bedoeld voor extra kosten van energie, koffie en slijtage "
            "van de werkruimte thuis. "
            "De vergoeding mag niet gecombineerd worden met een reiskostenvergoeding voor "
            "dezelfde werkdag. De werknemer kan kiezen: thuiswerkvergoeding óf reiskosten, "
            "niet beide op dezelfde dag."
        ),
    },
    {
        "doc_id": "IB2024-36a",
        "doc_type": "legislation",
        "doc_title": "Wet Inkomstenbelasting 2024",
        "article": "Artikel 3.6a",
        "paragraph": "Kinderopvangaftrek",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Artikel 3.6a — Kinderopvangtoeslag en aftrek. "
            "Ouders met kinderen tot 12 jaar kunnen kinderopvangtoeslag aanvragen via de "
            "Belastingdienst. De toeslag bedraagt maximaal 96% van de kosten afhankelijk van "
            "het toetsingsinkomen. "
            "Beide ouders moeten een arbeidsinkomen hebben of een erkende opleiding volgen. "
            "Kinderopvangtoeslag en persoonsgebonden aftrek voor kinderopvang kunnen niet "
            "tegelijkertijd worden geclaimd voor dezelfde kosten (verbod op dubbele aftrek)."
        ),
    },
    {
        "doc_id": "IB2024-box3",
        "doc_type": "legislation",
        "doc_title": "Wet Inkomstenbelasting 2024",
        "article": "Artikel 5.2",
        "paragraph": "Box 3 vermogensrendementsheffing",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Artikel 5.2 — Belastbaar inkomen uit sparen en beleggen (box 3) 2024. "
            "Het heffingsvrij vermogen bedraagt € 57.000 per belastingplichtige (fiscale "
            "partners: € 114.000 gezamenlijk). "
            "Forfaitair rendement: spaargeld 1,03%, overige beleggingen 6,04%, schulden -2,47%. "
            "Tarief box 3: 36%. "
            "Na het Kerstarrest (HR 24-12-2021, ECLI:NL:HR:2021:1963) geldt rechtsherstel "
            "voor jaren 2017-2022 op basis van werkelijk rendement."
        ),
    },
    {
        "doc_id": "ECLI-HR-2023-123",
        "doc_type": "case_law",
        "doc_title": "Hoge Raad 14 april 2023",
        "article": "ECLI:NL:HR:2023:123",
        "paragraph": "Thuiswerkkostenaftrek DGA",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Hoge Raad, 14 april 2023, ECLI:NL:HR:2023:123. "
            "Betreft: directeur-grootaandeelhouder (DGA) die werkruimte thuis claimt als "
            "aftrekpost. "
            "Overwegingen: De werkruimte moet voldoen aan het zelfstandigheidscriterium: "
            "een afsluitbare ruimte met eigen ingang of sanitair, hoofdzakelijk zakelijk "
            "gebruikt (>70%). Een slaapkamer die incidenteel als kantoor wordt gebruikt "
            "voldoet niet. "
            "Beslissing: Cassatie verworpen. Aftrek werkruimte thuis terecht geweigerd. "
            "De inspecteur heeft de aanslag terecht gehandhaafd."
        ),
    },
    {
        "doc_id": "ECLI-HR-2021-1963",
        "doc_type": "case_law",
        "doc_title": "Hoge Raad 24 december 2021 — Kerstarrest",
        "article": "ECLI:NL:HR:2021:1963",
        "paragraph": "Box 3 strijdig met eigendomsrecht",
        "classification": "public",
        "access_roles": ["*"],
        "text": (
            "Hoge Raad, 24 december 2021, ECLI:NL:HR:2021:1963 (Kerstarrest). "
            "Het box 3-stelsel zoals dat gold van 2017 tot 2022 is in strijd met "
            "artikel 1 van het Eerste Protocol bij het EVRM (eigendomsrecht) en "
            "artikel 14 EVRM (discriminatieverbod), voor zover het werkelijke rendement "
            "lager is dan het forfaitaire rendement. "
            "De Belastingdienst moet rechtsherstel bieden op basis van het werkelijke "
            "rendement. Belastingplichtigen die tijdig bezwaar hebben gemaakt, hebben "
            "recht op teruggaaf."
        ),
    },
    {
        "doc_id": "BELEIDSNOTITIE-2024-DGA",
        "doc_type": "policy",
        "doc_title": "Intern Beleid: Beoordeling DGA-dossiers 2024",
        "article": "Sectie 4.2",
        "paragraph": "Gebruikelijk loon",
        "classification": "restricted",
        "access_roles": ["restricted", "legal_classified", "fiod"],
        "text": (
            "Intern beleid sectie 4.2 — Gebruikelijk loon DGA (2024). "
            "Het gebruikelijk loon voor een DGA bedraagt minimaal € 56.000 in 2024 "
            "(was € 51.000 in 2023). "
            "Inspecteurs dienen bij controle te verifiëren of het loon marktconform is. "
            "Bij afwijking van meer dan 30% van het marktloon kan de inspecteur corrigeren. "
            "Aandachtspunt: DGA's in verliesgevende BV's mogen een lager gebruikelijk loon "
            "overeenkomen mits gedocumenteerd en goedgekeurd via vooroverleg."
        ),
    },
    {
        "doc_id": "FIOD-HANDLEIDING-2024-01",
        "doc_type": "policy",
        "doc_title": "FIOD Opsporingshandleiding — Belastingfraude 2024",
        "article": "Hoofdstuk 7",
        "paragraph": "Indicatoren omzetbelastingfraude",
        "classification": "fiod",
        "access_roles": ["fiod"],
        "text": (
            "FIOD Opsporingshandleiding 2024, Hoofdstuk 7 — Indicatoren belastingfraude. "
            "Rode vlaggen voor BTW-carrouselfraude: hoge omzet met lage marges, "
            "frequente wisselingen van bestuurders, transacties met entiteiten in "
            "hoogrisico-jurisdicties, ontbrekende fysieke voorraad bij papieren leveringen. "
            "Opsporingsdrempel: aangifte-afwijkingen > € 50.000 of patroon van correcties "
            "over meerdere tijdvakken. "
            "Interne classificatie: STRIKT VERTROUWELIJK — uitsluitend voor FIOD-medewerkers."
        ),
    },
    {
        "doc_id": "ELEARNING-BTW-2024",
        "doc_type": "elearning",
        "doc_title": "E-learning module: BTW-grondslagen voor helpdesk",
        "article": "Module 3",
        "paragraph": "Btw-tarieven en vrijstellingen",
        "classification": "internal",
        "access_roles": ["internal"],
        "text": (
            "E-learning Module 3 — BTW-tarieven en vrijstellingen (2024). "
            "Algemeen tarief: 21%. "
            "Verlaagd tarief (9%): voedingsmiddelen, geneesmiddelen, boeken, personenvervoer, "
            "arbeidsintensieve diensten (kappers, schoenmakers). "
            "Vrijgesteld (0% met recht op aftrek voorbelasting): export buiten EU, "
            "intracommunautaire leveringen. "
            "Vrijgesteld zonder aftrekrecht: medische diensten, onderwijs, financiële "
            "diensten, verhuur onroerend goed (tenzij geopteerd voor belaste verhuur). "
            "Helpdesk-instructie: verwijs bij twijfel over tarief altijd naar de "
            "Tabel I en Tabel II bij de Wet OB 1968."
        ),
    },
]
