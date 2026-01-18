# C Gamepad Recorder

Versions C du pad recorder pour déboguer les problèmes de compatibilité gamepad sous Windows.

## Trois implémentations

Ce dossier contient trois versions différentes utilisant différentes APIs Windows :

### 1. DirectInput (`pad_recorder_dinput.c`)
- **API**: DirectInput 8
- **Usage**: API classique pour gamepads génériques
- **Avantages**: Compatible avec la plupart des gamepads, y compris les pads de danse DDR
- **Inconvénients**: API ancienne mais toujours supportée

### 2. XInput (`pad_recorder_xinput.c`)
- **API**: XInput
- **Usage**: API moderne pour contrôleurs Xbox
- **Avantages**: Simple, bien supporté pour les manettes Xbox
- **Inconvénients**: **Seulement pour contrôleurs Xbox 360/One/Series**, ne fonctionne pas avec d'autres gamepads

### 3. Raw HID (`pad_recorder_rawhid.c`)
- **API**: Raw Input HID
- **Usage**: Accès direct aux périphériques HID
- **Avantages**: Niveau le plus bas, voit tous les événements HID
- **Inconvénients**: Plus complexe, peut capturer plus de boutons que nécessaire

## Compilation

### Prérequis
- MinGW-w64 ou MSYS2 avec gcc pour Windows
- Les bibliothèques Windows SDK (normalement incluses avec MinGW)

### Compiler toutes les versions
```bash
make all
```

### Compiler individuellement
```bash
# DirectInput
make pad_recorder_dinput.exe

# XInput
make pad_recorder_xinput.exe

# Raw HID
make pad_recorder_rawhid.exe
```

### Compilation manuelle (sans make)
```bash
# DirectInput
gcc -Wall -O2 pad_recorder_dinput.c -o pad_recorder_dinput.exe -ldinput8 -ldxguid -lole32 -loleaut32

# XInput
gcc -Wall -O2 pad_recorder_xinput.c -o pad_recorder_xinput.exe -lxinput

# Raw HID
gcc -Wall -O2 pad_recorder_rawhid.c -o pad_recorder_rawhid.exe -lhid -lsetupapi
```

## Utilisation

1. Brancher votre gamepad
2. Vérifier qu'il est visible dans `joy.cpl` (Panneau de configuration > Périphériques de jeu)
3. Lancer un des exécutables :

```bash
# Essayer DirectInput en premier (le plus compatible)
./pad_recorder_dinput.exe

# Si le pad n'est pas détecté, essayer XInput (seulement pour Xbox)
./pad_recorder_xinput.exe

# Si rien ne fonctionne, essayer Raw HID
./pad_recorder_rawhid.exe
```

4. Appuyer sur les boutons du gamepad
5. Les événements s'affichent dans le terminal
6. Les données sont enregistrées dans un fichier CSV : `YYYY_MM_DD_HH_MM_SS_pad_record_[api].csv`
7. Arrêter avec `Ctrl+C`

## Diagnostique

### Le gamepad n'est pas détecté
- **DirectInput**: Le gamepad n'est pas configuré comme périphérique de jeu Windows
- **XInput**: Le gamepad n'est pas un contrôleur Xbox (très courant pour les pads de danse)
- **Raw HID**: Vérifier les permissions et que le périphérique est bien HID

### Visible dans joy.cpl mais pas de boutons détectés
Cela peut indiquer :
1. Le pilote ne transmet pas correctement les événements
2. Le gamepad nécessite une configuration spéciale
3. Le mapping des boutons est différent (essayer d'appuyer sur différents boutons)

### Que faire si aucune version ne fonctionne
1. Vérifier les pilotes du gamepad
2. Tester avec un autre programme (par ex. jstest sous Linux, ou joy.cpl)
3. Le pad pourrait nécessiter un pilote spécifique ou un mode de compatibilité

## Format de sortie CSV

```
timestamp,button,pressed
0.123,LEFT,1
0.456,LEFT,0
1.234,RIGHT,1
```

- `timestamp`: Temps écoulé en secondes depuis le démarrage
- `button`: Nom du bouton (LEFT, RIGHT, UP, DOWN, ou BTN0, BTN1, etc.)
- `pressed`: 1 pour appuyé, 0 pour relâché

## Notes

- Les trois versions enregistrent les 4 premiers boutons
- Raw HID peut enregistrer plus de boutons si le gamepad en a
- Les fichiers CSV sont compatibles avec la version Python
- Testé sur Windows 10/11 avec MinGW-w64
