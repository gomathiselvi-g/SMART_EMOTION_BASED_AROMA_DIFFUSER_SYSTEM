#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Smart Aroma timer firmware. Serial protocol: H,5\n / S,5\n / N,5\n / X,5\n
LiquidCrystal_I2C lcd(0x27, 16, 2);
const byte RELAY_PINS[4] = {2, 3, 4, 5};
const char RELAY_CODES[4] = {'H', 'S', 'N', 'X'};
const char* AROMA_NAMES[4] = {"Citrus", "Lavender", "Rosemary", "Eucalyptus"};
const bool RELAY_ACTIVE_LOW = false;  // Change to true only if your relay board is active-LOW.
const unsigned long MAX_ON_MS = 30000UL;

int activeRelay = -1;
unsigned long relayOffAt = 0;
String inputLine;

void setRelay(int index, bool on) {
  if (index < 0 || index >= 4) return;
  const byte offLevel = RELAY_ACTIVE_LOW ? HIGH : LOW;
  const byte onLevel = RELAY_ACTIVE_LOW ? LOW : HIGH;
  digitalWrite(RELAY_PINS[index], on ? onLevel : offLevel);
}

void stopRelay() {
  if (activeRelay >= 0) setRelay(activeRelay, false);
  activeRelay = -1;
  relayOffAt = 0;
  lcd.clear(); lcd.setCursor(0, 0); lcd.print("Smart Aroma");
  lcd.setCursor(0, 1); lcd.print("Ready");
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;
  int comma = line.indexOf(',');
  if (comma < 0) return; // Ignore old/invalid commands safely.
  char code = line.substring(0, comma).charAt(0);
  long seconds = line.substring(comma + 1).toInt();
  if (seconds < 1) seconds = 1;
  if (seconds > 30) seconds = 30;
  int idx = -1;
  for (int i = 0; i < 4; ++i) if (code == RELAY_CODES[i]) idx = i;
  if (idx < 0) return;

  // One relay at a time: switch any prior relay off before starting a new pulse.
  stopRelay();
  activeRelay = idx;
  setRelay(activeRelay, true);
  relayOffAt = millis() + (unsigned long)seconds * 1000UL;
  lcd.clear(); lcd.setCursor(0, 0); lcd.print(AROMA_NAMES[idx]);
  lcd.setCursor(0, 1); lcd.print("Spray "); lcd.print(seconds); lcd.print(" sec");
}

void setup() {
  Serial.begin(9600);
  lcd.init(); lcd.backlight();
  for (byte i = 0; i < 4; ++i) { pinMode(RELAY_PINS[i], OUTPUT); setRelay(i, false); }
  stopRelay();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') { handleCommand(inputLine); inputLine = ""; }
    else if (c != '\r' && inputLine.length() < 40) inputLine += c;
    else if (inputLine.length() >= 40) inputLine = "";
  }
  if (activeRelay >= 0 && (long)(millis() - relayOffAt) >= 0) stopRelay();
}
