import os
import playerposprediction
import playerrecommender


def main():
    bol = False
    while True:
        if bol:
            string = "Vuoi che ti suggerisca un altro giocatore? - Premi 1\n" \
                     + "Vuoi sapere il ruolo di un altro giocatore? - Premi 2\n" \
                     + "Vuoi uscire? - Premi 3\n"
        else:
            string = "Ciao, cosa vuoi fare?\n" + "Vuoi che ti suggerisca un giocatore? - Premi 1\n" \
                     + "Vuoi sapere il ruolo di un giocatore? - Premi 2\n" \
                     + "Vuoi uscire? - Premi 3\n"
        response = input(string)

        if response == '1':
            # RACCOMANDAZIONE DI GIOCATORI

            pos = ''
            matchplayed = ''
            gls = ''
            ast = ''
            crdy = ''
            crdr = ''
            born = ''

            print("Ti farò delle domande per poterti suggerire un giocatore.")

            # Nome del giocatore
            player = input("Suggeriscimi il nome di un giocatore che ti piace\n").lower()

            # Ruolo del giocatore
            string = input("Sai dirmi in che ruolo gioca questo giocatore? (Sì o No)" +
                           "(se non sai i ruoli esistenti digita 0)\n").lower()
            while string == '0':
                print("I differenti ruoli nel calcio sono:\n" +
                      "- GK : il giocatore è un portiere\n" +
                      "- DF : il giocatore è un difensore (terzino, difensore centrale)\n" +
                      "- MF : il giocatore è un centrocampista (mediano, esterno, trequartista, mezz'ala)\n" +
                      "- FW : il giocatore è un attaccante (punta, falso nove, ala)")
                string = input("Ora sai dirmi in che ruolo gioca questo giocatore? (Sì o No)" +
                               "(se non sai i ruoli esistenti digita 0)\n").lower()
            if string == "si":
                pos = input("In che ruolo gioca questo giocatore?\n").lower()

            # Anno di nascita del giocatore
            string = input("Sai dirmi in che anno è nato questo giocatore? (Sì o No)\n").lower()
            if string == "si":
                born = input("In che anno è nato?\n").lower()

            # Numero di partite giocate dal giocatore
            string = input("Sai dirmi quante partite ha giocato questo giocatore? (Sì o No)" +
                           "(se non sai come calcolare il numero di partite giocate digita 0)\n").lower()
            while string == '0':
                print("Dividi i minuti totali giocati per 90 e arrotonda il risultato ottenuto!\n")
                string = input("Ora sai dirmi quante partite ha giocato questo giocatore? (Sì o No)" +
                               "(se non sai cos'è il tipo digita 0)\n").lower()
            if string == "si":
                matchplayed = input("Quante partite ha giocato?\n").lower()

            # Numero di goal segnati dal giocatore
            string = input("Sai dirmi quanti goal ha segnato questo giocatore? (Sì o No)\n").lower()
            if string == "si":
                gls = input("Quanti goal ha segnato?\n").lower()

            # Numero di assist fatti dal giocatore
            string = input("Sai dirmi quanti assist ha fatto questo giocatore? (Sì o No)\n").lower()
            if string == "si":
                ast = input("Quanti assist ha fatto?\n").lower()

            # Numero di cartellini gialli presi dal giocatore
            string = input("Sai dirmi quanti cartellini gialli ha preso questo giocatore? (Sì o No)\n").lower()
            if string == "si":
                crdy = input("Quanti cartellini gialli ha preso?\n").lower()

            # Numero di cartellini rossi presi dal giocatore
            string = input("Sai dirmi quanti cartellini rossi ha preso questo giocatore? (Sì o No)\n").lower()
            if string == "si":
                crdr = input("Quanti cartellini rossi ha preso?\n").lower()

            # Chiama la funzione di raccomandazione con i dati forniti
            playerrecommender.main(ast, pos, matchplayed, crdy, crdr, gls, player, born)
            print("\n")
            os.system("pause")
            bol = True
            print("\n")

        elif response == '2':
            # PREDIZIONE DEL RUOLO DEL GIOCATORE

            print("Ti chiederò un po' di cose. Iniziamo...")

            # Nome del giocatore da classificare
            player = input("Qual è il nome del giocatore che vuoi classificare?\n ").lower()

            # Anno di nascita del giocatore
            born = input("In che anno è nato il giocatore che vuoi classificare?\n").lower()

            # Numero di partite giocate dal giocatore da classificare
            matchplayed = input("Quante partite ha giocato il giocatore che vuoi classificare? " +
                                "(se non sai come calcolare il numero di partite giocate digita 0)\n").lower()

            while matchplayed == '0':
                print("Dividi i minuti totali giocati per 90 e arrotonda il risultato ottenuto!")
                matchplayed = input("Quindi, quante partite ha giocato il giocatore che vuoi classificare? " +
                                    "(se non sai come calcolare il numero di partite giocate digita 0)\n").lower()

            # Numero di goal segnati dal giocatore da classificare
            gls = input("Quanti goal ha segnato il giocatore che vuoi classificare?\n").lower()

            # Numero di assist fatti dal giocatore da classificare
            ast = input("Quanti assist ha fatto il giocatore che vuoi classificare?\n").lower()

            # Numero di cartellini gialli presi dal giocatore da classificare
            crdy = input("Quanti cartellini gialli ha preso il giocatore che vuoi classificare?\n").lower()

            # Numero di cartellini rossi presi dal giocatore da classificare
            crdr = input("Quanti cartellini rossi ha preso il giocatore che vuoi classificare?\n").lower()

            # Chiama la funzione di predizione del ruolo con i dati forniti
            playerposprediction.predict_pos(player, born, matchplayed, gls, ast, crdy, crdr)
            print("\n")
            os.system("pause")
            bol = True
            print("\n\n")

        else:
            # Uscita dal programma
            print("Arrivederci!")
            break


if __name__ == '__main__':
    main()
