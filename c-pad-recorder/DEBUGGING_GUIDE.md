# Guide de Débogage Rapide

## Problème: Le pad recorder Python ne fonctionne pas

### Étapes de diagnostic

1. **Vérifier la visibilité du gamepad**
   ```
   - Ouvrir joy.cpl (Panneau de configuration > Périphériques de jeu)
   - Le gamepad est-il listé ?
   - Les boutons s'affichent-ils dans joy.cpl ?
   ```

2. **Tester avec les versions C**

   **Commencer par DirectInput** (le plus compatible):
   ```bash
   cd c-pad-recorder
   make pad_recorder_dinput.exe
   ./pad_recorder_dinput.exe
   ```
   
   Si ça fonctionne → Le problème vient de pygame
   Si ça ne fonctionne pas → Continuer...

   **Essayer XInput** (seulement Xbox):
   ```bash
   make pad_recorder_xinput.exe
   ./pad_recorder_xinput.exe
   ```
   
   Message "No XInput controller found" = Le pad n'est pas Xbox compatible (normal pour pads DDR)
   
   **Essayer Raw HID** (bas niveau):
   ```bash
   make pad_recorder_rawhid.exe
   ./pad_recorder_rawhid.exe
   ```
   
   Si ça fonctionne → Le pad utilise HID mais pas DirectInput/XInput
   Si ça ne fonctionne pas → Problème de pilote

3. **Analyser les résultats**

   | DirectInput | XInput | Raw HID | Diagnostic |
   |-------------|--------|---------|------------|
   | ✓ | ✗ | ✓ | Pad générique, problème pygame |
   | ✗ | ✓ | ✓ | Pad Xbox, pygame mal configuré |
   | ✗ | ✗ | ✓ | Pad HID pur, besoin Raw Input |
   | ✗ | ✗ | ✗ | Problème pilote/matériel |

## Solutions courantes

### Pygame ne voit pas le gamepad
```python
# Vérifier la version de pygame
import pygame
print(pygame.version.ver)

# Lister les joysticks détectés
pygame.init()
pygame.joystick.init()
print(f"Joysticks: {pygame.joystick.get_count()}")
```

### Le gamepad nécessite un pilote spécifique
- Certains pads DDR ont besoin de pilotes custom
- Chercher "votre_modèle_pad windows driver"
- Parfois il faut un émulateur (comme vJoy ou x360ce)

### Le pad fonctionne en C mais pas en Python
→ Le problème est dans pygame
→ Solutions:
  1. Mettre à jour pygame: `pip install --upgrade pygame`
  2. Essayer une version SDL2: `pip install pygame==2.0.0+`
  3. Utiliser une bibliothèque alternative (inputs, evdev sous Linux)

## Pour aller plus loin

Si aucune version C ne fonctionne:
1. Vérifier dans le Gestionnaire de périphériques Windows
2. Chercher des messages d'erreur dans l'Observateur d'événements
3. Tester sur un autre PC pour isoler le problème
4. Le pad pourrait être défectueux

Si Raw HID fonctionne uniquement:
- Adapter le code Python pour utiliser Raw Input
- Ou utiliser la version C comme solution de contournement
