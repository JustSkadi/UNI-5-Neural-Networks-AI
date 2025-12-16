% ZADANIE 1

malzenstwo(jan, maria).
malzenstwo(piotr, anna).
malzenstwo(tomasz, katarzyna).
malzenstwo(adam, agnieszka).
malzenstwo(pawel, julia).
malzenstwo(marcin, barbara).

dziecko(piotr, jan).
dziecko(piotr, maria).
dziecko(tomasz, jan).
dziecko(tomasz, maria).
dziecko(marcin, jan).
dziecko(marcin, maria).
dziecko(adam, piotr).
dziecko(adam, anna).
dziecko(agnieszka, tomasz).
dziecko(agnieszka, katarzyna).
dziecko(pawel, piotr).
dziecko(pawel, anna).
dziecko(julia, tomasz).
dziecko(julia, katarzyna).
dziecko(ewa, marcin).
dziecko(ewa, barbara).
dziecko(robert, marcin).
dziecko(robert, barbara).

kobieta(maria).
kobieta(anna).
kobieta(katarzyna).
kobieta(agnieszka).
kobieta(julia).
kobieta(barbara).
kobieta(ewa).

% ZADANIE 2

mezczyzna(X) :- \+ kobieta(X), osoba(X).

osoba(X) :- malzenstwo(X, _).
osoba(X) :- malzenstwo(_, X).
osoba(X) :- dziecko(X, _).
osoba(X) :- dziecko(_, X).

rodzic(X, Y) :- dziecko(Y, X).

ojciec(X, Y) :- rodzic(X, Y), mezczyzna(X).
matka(X, Y) :- rodzic(X, Y), kobieta(X).

dziadek(X, Y) :- rodzic(X, Z), rodzic(Z, Y), mezczyzna(X).
babcia(X, Y) :- rodzic(X, Z), rodzic(Z, Y), kobieta(X).

pradziadek(X, Y) :- rodzic(X, Z), dziadek(Z, Y), mezczyzna(X).
prababcia(X, Y) :- rodzic(X, Z), babcia(Z, Y), kobieta(X).

syn(X, Y) :- dziecko(X, Y), mezczyzna(X).
syn(X) :- syn(X, Y).
corka(X, Y) :- dziecko(X, Y), kobieta(X).

brat(X, Y) :- dziecko(X, Z), dziecko(Y, Z), mezczyzna(X), X \= Y.
siostra(X, Y) :- dziecko(X, Z), dziecko(Y, Z), kobieta(X), X \= Y.

wuj(X, Y) :- brat(X, Z), rodzic(Z, Y), mezczyzna(X).
ciotka(X, Y) :- siostra(X, Z), rodzic(Z, Y), kobieta(X).

kuzyn(X, Y) :- dziecko(X, Z), dziecko(Y, W), brat(Z, W), mezczyzna(X).
kuzynka(X, Y) :- dziecko(X, Z), dziecko(Y, W), siostra(Z, W), kobieta(X).
niejestw(X, L) :- \+ member(X, L).
malzonek(X, Y) :- malzenstwo(X, Y).
malzonek(X, Y) :- malzenstwo(Y, X).

szwagier(X, Y) :- malzonek(X, Z), brat(Z, Y), mezczyzna(X).
bratowa(X, Y) :- malzonek(X, Z), brat(Z, Y), kobieta(X).

ziec(X, Y) :- malzonek(X, Z), dziecko(Z, Y), mezczyzna(X).
synowa(X, Y) :- malzonek(X, Z), dziecko(Z, Y), kobieta(X).

dziadek(L) :- 
    wszystkie_osoby(Wszystkie),
    filtr_dziadkow(Wszystkie, L).

dodaj(L, M) :- osoba(X), niejestw(X, L), dodaj([X|L], M), !.
dodaj(M, M).
wszystkie_osoby(L) :- dodaj([], L).

dodaj_syn(L, M) :- syn(X), niejestw(X, L), dodaj_syn([X|L], M), !.
dodaj_syn(M, M).
wszyscy_synowie(L) :- dodaj_syn([], L).

filtr_dziadkow([], []).
filtr_dziadkow([H|T], [H|Wynik]) :-
    dziadek(H, _),
    !,
    filtr_dziadkow(T, Wynik).
filtr_dziadkow([_|T], Wynik) :-
    filtr_dziadkow(T, Wynik).

% ZADANIE 3

dlugosc(Xs,L) :- dlugosc(Xs,0,L) .
dlugosc( []     , L , L ) .
dlugosc( [_|Xs] , T , L ) :-
  T1 is T+1 ,
  dlugosc(Xs,T1,L)
  .

przodek(X, Y) :- rodzic(X, Y).
przodek(X, Y) :- rodzic(X, Z), przodek(Z, Y).

potomek(X, Y) :- przodek(Y, X).

lista_przodkow(X, L) :- 
    wszystkie_osoby(Wszystkie),
    filtr_przodkow(X, Wszystkie, L).

filtr_przodkow(_, [], []).
filtr_przodkow(X, [H|T], [H|Wynik]) :-
    przodek(H, X),
    !,
    filtr_przodkow(X, T, Wynik).
filtr_przodkow(X, [_|T], Wynik) :-
    filtr_przodkow(X, T, Wynik).

lista_potomkow(X, L) :- 
    wszystkie_osoby(Wszystkie),
    filtr_potomkow(X, Wszystkie, L).

filtr_potomkow(_, [], []).
filtr_potomkow(X, [H|T], [H|Wynik]) :-
    potomek(H, X),
    !,
    filtr_potomkow(X, T, Wynik).
filtr_potomkow(X, [_|T], Wynik) :-
    filtr_potomkow(X, T, Wynik).

liczba_przodkow(X) :- lista_przodkow(X, L), dlugosc(L, _).
liczba_potomkow(X) :- lista_potomkow(X, L), dlugosc(L, _).

% ZADANIE 4

lista_potomkow_z_relacjami(X, S) :- 
    lista_potomkow(X, L), 
    opis_potomkow(X, L, S).

opis_potomkow(_, [], '').
opis_potomkow(X, [Y], S) :- 
    znajdz_relacje(X, Y, Relacja),
    format(atom(S), '~w (~w)', [Y, Relacja]).
opis_potomkow(X, [Y|Rest], S) :- 
    Rest = [_|_],
    znajdz_relacje(X, Y, Relacja),
    opis_potomkow(X, Rest, RestS),
    format(atom(S), '~w (~w), ~w', [Y, Relacja, RestS]).

znajdz_relacje(X, Y, syn) :- syn(Y, X), !.
znajdz_relacje(X, Y, corka) :- corka(Y, X), !.
znajdz_relacje(X, Y, synowa) :- synowa(Y, X), !.
znajdz_relacje(X, Y, ziec) :- ziec(Y, X), !.
znajdz_relacje(_, _, potomek).

% ZADANIE 5 vol 1

zadanie5(L) :- 
    wszystkie_osoby(Wszystkie),
    filtr_z_potomkami(Wszystkie, L).

filtr_z_potomkami([], []).
filtr_z_potomkami([H|T], [H|Wynik]) :-
    potomek(_, H),
    !,
    filtr_z_potomkami(T, Wynik).
filtr_z_potomkami([_|T], Wynik) :-
    filtr_z_potomkami(T, Wynik).

wynik5(X) :- zadanie5(L), dlugosc(L, X).

% ZADANIE 5 vol 2

zadanie5_2(X) :- osoba(X), \+ potomek(_, X).


