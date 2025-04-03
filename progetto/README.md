# cartella progetto

il progetto creato è all'interno della cartella progetto. il resto è il contenuto creato a lezione.

# descrizione contenuto e struttura dataset

il dataset descrive il prezzo di vendita una raccolta di case, con altre 6 colonne che descrivono le loro caratteristiche principali come posizione (latitudine e longitudine), data dell'acquisto, tempo dalla data di costruzione, distanza dalla metropolitana più vicina e numero di minimarket vicini.

# descrizione modello

i modelli utilizzati sono dei random forest con determinate covariate a determinare le previsioni. il primo descrive i prezzi delle case in base alla posizione (latitudine e longitudine), mentre il secondo è costruito ponendo come covariate l'anno di costruzione, la distanza minima dalla metro e il numero di minimarket vicini.

# struttura applicazione

il codice alla base (pipeline.py) costruisce i 2 modelli descritti in precedenza sfruttando funzioni create precedentemente su altri codici. l'interfaccia utente è costruita in modo da permettere la scelta del modello e in base a questa è possibile inserire i valori delle variabili richieste.
