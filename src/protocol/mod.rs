use serde::{Deserialize, Serialize};
use crate::{Result, OpenSmellError};

/// Semantic kind of a streamed channel. The default is a classic analog MOX
/// cell; MEMS digital chips are I²C and should be treated as a raw index, env
/// values come from DHT-style telemetry, and `Fan` is a control actuator.
/// Kinds are declared by firmware with an `OSMK` line; absent that, everything
/// is assumed `AnalogMox` (backwards-compatible with v1 CSV streams).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelKind {
    AnalogMox,
    MemsIndex,
    EnvTemp,
    EnvHum,
    Fan,
}

impl ChannelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelKind::AnalogMox => "analog_mox",
            ChannelKind::MemsIndex => "mems_index",
            ChannelKind::EnvTemp => "env_temp",
            ChannelKind::EnvHum => "env_hum",
            ChannelKind::Fan => "fan",
        }
    }

    pub fn parse(s: &str) -> Option<ChannelKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "analog" | "analog_mox" | "mox" | "mq" => Some(ChannelKind::AnalogMox),
            "mems" | "mems_index" | "digital" => Some(ChannelKind::MemsIndex),
            "temp" | "env_temp" | "temperature" => Some(ChannelKind::EnvTemp),
            "hum" | "env_hum" | "humidity" => Some(ChannelKind::EnvHum),
            "fan" | "rpm" => Some(ChannelKind::Fan),
            _ => None,
        }
    }
}

/// OSM protocol message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OsmMessage {
    /// Sensor data: OSM,<adc0>,<adc1>,...,<adcN>
    Data { channels: Vec<f64>, timestamp: f64 },
    /// Device info: INFO,<device_id>,<firmware_version>,<n_sensors>
    Info { device_id: String, firmware_version: String, n_sensors: usize },
    /// Calibration request: CAL,<channel>,<r0_value>
    Calibration { channel: usize, r0_value: f64 },
    /// Phase boundary: EVENT,<label>,<sample_index>[,t_ms=<MS>]
    /// `sample_index` is the index of the first sample of the new phase, so the
    /// next `Data` line is that sample. `t_ms` is the host's monotonic clock at
    /// injection, when the sender supplied one.
    Event { label: String, sample_index: i64, t_ms: Option<i64> },
    /// Error message: ERR,<error_code>,<message>
    Error { code: i32, message: String },
    /// Heartbeat: PING
    Ping,
    /// Per-column channel kinds: OSMK,<kind0>,<kind1>,...,<kindN>
    /// e.g. OSMK,analog,mems,env_temp,env_hum,fan
    Kinds { kinds: Vec<ChannelKind> },
    /// Environmental telemetry (DHT-style): ENV,<temp_C>,<hum_pct>
    Env { temperature: f64, humidity: f64 },
    /// Unknown line.
    Unknown(String),
}

/// Conventional `EVENT` labels. Free-form on the wire — an unrecognised label is
/// a legal phase name — but these are what the recording and submission layers
/// recognise.
pub const KNOWN_PHASE_LABELS: [&str; 6] = [
    "baseline", "exposure", "exposure_a", "exposure_b", "gap", "recovery",
];

/// Build the line a host sends to mark a phase boundary.
///
/// `sample_index` is the index of the first sample of the new phase; pass
/// [`SAMPLE_INDEX_DEVICE_ASSIGNED`] to have the device substitute its own
/// next-sample index, which is what a host that is not counting should do.
/// `t_ms` is the caller's monotonic clock at injection and is optional.
///
/// The result carries no trailing newline — the caller appends one, as with
/// `parse_line` taking a line without one. The label must not contain a comma:
/// the wire is comma-delimited, and a comma inside the label would move every
/// following field. A malformed line is rejected by the device with `ERR,8`
/// rather than silently reinterpreted.
pub fn format_event(label: &str, sample_index: i64, t_ms: Option<i64>) -> String {
    let mut line = format!("EVENT,{},{}", label, sample_index);
    if let Some(t) = t_ms {
        line.push_str(&format!(",t_ms={}", t));
    }
    line
}

/// `sample_index` value asking the device to assign the boundary itself.
pub const SAMPLE_INDEX_DEVICE_ASSIGNED: i64 = -1;

/// OSM protocol parser.
/// Works with any MCU (ESP32, Arduino, STM32, RPi) that sends CSV-like data.
pub struct OsmProtocol {
    /// Expected number of channels (0 = auto-detect).
    expected_channels: usize,
    /// Timestamp offset (for syncing device time with host time).
    timestamp_offset: f64,
}

impl OsmProtocol {
    pub fn new(expected_channels: usize) -> Self {
        Self {
            expected_channels,
            timestamp_offset: 0.0,
        }
    }

    /// Parse a single line from the device.
    pub fn parse_line(&self, line: &str, host_timestamp: f64) -> Result<OsmMessage> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(OsmMessage::Unknown(line.to_string()));
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            return Ok(OsmMessage::Unknown(line.to_string()));
        }

        match parts[0].to_uppercase().as_str() {
            "OSM" => self.parse_data(&parts[1..], host_timestamp),
            "OSMK" => self.parse_kinds(&parts[1..]),
            "ENV" => self.parse_env(&parts[1..]),
            "INFO" => self.parse_info(&parts[1..]),
            "CAL" => self.parse_calibration(&parts[1..]),
            "EVENT" => self.parse_event(&parts[1..]),
            "ERR" => self.parse_error(&parts[1..]),
            "PING" => Ok(OsmMessage::Ping),
            _ => Ok(OsmMessage::Unknown(line.to_string())),
        }
    }

    /// OSMK,<kind0>,...,<kindN> — a per-column kind declaration. Unknown tokens
    /// fall back to `AnalogMox` (lenient) so a slightly-off declaration never
    /// kills the stream; the desktop dataset stays honest regardless.
    fn parse_kinds(&self, parts: &[&str]) -> Result<OsmMessage> {
        let kinds: Vec<ChannelKind> = parts.iter()
            .map(|p| ChannelKind::parse(p).unwrap_or(ChannelKind::AnalogMox))
            .collect();
        Ok(OsmMessage::Kinds { kinds })
    }

    /// ENV,<temp_C>,<hum_pct> — environmental telemetry.
    fn parse_env(&self, parts: &[&str]) -> Result<OsmMessage> {
        if parts.len() < 2 {
            return Err(OpenSmellError::FeatureExtraction("ENV message too short".to_string()));
        }
        let temperature = parts[0].trim().parse::<f64>().unwrap_or(f64::NAN);
        let humidity = parts[1].trim().parse::<f64>().unwrap_or(f64::NAN);
        Ok(OsmMessage::Env { temperature, humidity })
    }

    fn parse_data(&self, parts: &[&str], host_timestamp: f64) -> Result<OsmMessage> {
        let channels: Result<Vec<f64>> = parts
            .iter()
            .map(|p| {
                p.trim().parse::<f64>()
                    .map_err(|_| OpenSmellError::FeatureExtraction(
                        format!("Invalid float: {}", p)
                    ))
            })
            .collect();
        let channels = channels?;

        // Auto-pad or trim to expected channel count
        let channels = if self.expected_channels > 0 {
            let mut ch = channels;
            while ch.len() < self.expected_channels {
                ch.push(0.0);
            }
            ch.truncate(self.expected_channels);
            ch
        } else {
            channels
        };

        Ok(OsmMessage::Data {
            channels,
            timestamp: host_timestamp + self.timestamp_offset,
        })
    }

    fn parse_info(&self, parts: &[&str]) -> Result<OsmMessage> {
        if parts.len() < 3 {
            return Err(OpenSmellError::FeatureExtraction("INFO message too short".to_string()));
        }
        Ok(OsmMessage::Info {
            device_id: parts[0].trim().to_string(),
            firmware_version: parts[1].trim().to_string(),
            n_sensors: parts[2].trim().parse().unwrap_or(0),
        })
    }

    fn parse_calibration(&self, parts: &[&str]) -> Result<OsmMessage> {
        if parts.len() < 2 {
            return Err(OpenSmellError::FeatureExtraction("CAL message too short".to_string()));
        }
        Ok(OsmMessage::Calibration {
            channel: parts[0].trim().parse().unwrap_or(0),
            r0_value: parts[1].trim().parse().unwrap_or(0.0),
        })
    }

    /// EVENT,<label>,<sample_index>[,t_ms=<MS>] — a phase boundary. `t_ms` is
    /// optional and may appear anywhere after the index; anything unparseable in
    /// it is dropped rather than failing the boundary, since the boundary itself
    /// is the part a reader cannot recover on its own.
    fn parse_event(&self, parts: &[&str]) -> Result<OsmMessage> {
        if parts.len() < 2 {
            return Err(OpenSmellError::FeatureExtraction("EVENT message too short".to_string()));
        }
        let label = parts[0].trim();
        if label.is_empty() {
            return Err(OpenSmellError::FeatureExtraction(
                "EVENT message has an empty label".to_string(),
            ));
        }
        let sample_index = parts[1].trim().parse().unwrap_or(SAMPLE_INDEX_DEVICE_ASSIGNED);
        let t_ms = parts[2..].iter().find_map(|p| {
            p.trim().strip_prefix("t_ms=").and_then(|v| v.trim().parse().ok())
        });
        Ok(OsmMessage::Event {
            label: label.to_string(),
            sample_index,
            t_ms,
        })
    }

    fn parse_error(&self, parts: &[&str]) -> Result<OsmMessage> {
        if parts.len() < 2 {
            return Ok(OsmMessage::Error { code: -1, message: "Unknown error".to_string() });
        }
        Ok(OsmMessage::Error {
            code: parts[0].trim().parse().unwrap_or(-1),
            message: parts[1..].join(","),
        })
    }
}

/// Generate Arduino firmware sketch for ESP32 with configurable sensor pins.
///
/// The generated firmware:
/// - connects to a WiFi network (or falls back to SoftAP mode),
/// - advertises an mDNS service `_osmograph._tcp` on TCP port 8080,
/// - runs a multi-client TCP server speaking the OSM protocol,
/// - streams `OSM` readings at 10 Hz over serial and to all clients,
/// - responds to `PING` (with `PONG`) and `CAL` (re-baseline),
/// - sends `INFO` with the declared cadence on every client connect,
/// - accepts `EVENT` phase boundaries and re-announces them on every output,
/// - collects a baseline (`r0` per channel) at boot.
pub fn generate_arduino_sketch(
    sensor_pins: &[u8],
    wifi_ssid: &str,
    wifi_password: &str,
) -> String {
    let pins_str = sensor_pins.iter()
        .map(|p| format!("  {}", p))
        .collect::<Vec<_>>()
        .join(",\n");
    let n_sensors = sensor_pins.len();

    format!(r#"
// OpenSmell ESP32 Firmware
// Generated by opensmell-rs protocol::generate_arduino_sketch
// Sensor pins: {sensor_pins:?}

#include <WiFi.h>
#include <WiFiServer.h>
#include <mDNS.h>
#include <string.h>

#define FW_VERSION "1.2.0"
#define OSM_SERVICE "_osmograph"
#define OSM_TCP_PORT 8080
#define SAMPLE_INTERVAL_MS 100          // 10 Hz per channel
#define BASELINE_SECONDS 30
#define BASELINE_SAMPLES (BASELINE_SECONDS * 10)
#define MAX_CLIENTS 4
#define AP_SSID_PREFIX "Osmograph-"

// WiFi credentials (empty SSID => SoftAP mode)
const char* WIFI_SSID = "{wifi_ssid}";
const char* WIFI_PASS = "{wifi_password}";

// Sensor configuration
const int SENSOR_PINS[] = {{
{pins_str}
}};
const int N_SENSORS = {n_sensors};

WiFiServer server(OSM_TCP_PORT);
WiFiClient clients[MAX_CLIENTS];

// Baseline (R0) values, updated during calibration
float r0[N_SENSORS] = {{0}};
bool calibrated = false;
uint32_t lastSampleMs = 0;

// Phase boundaries. A boundary is announced on every output immediately before
// the first OSM line of the new phase, so a reader knows which sample the label
// belongs to without counting. `pendingEvent` holds the boundary a host has
// injected but that has not reached the stream yet; SAMPLE_INDEX_ASSIGN asks the
// device to use its own counter, which is what a host that is not counting sends.
#define SAMPLE_INDEX_ASSIGN (-1L)
#define EVENT_LABEL_MAX 32
bool pendingEvent = false;
char pendingLabel[EVENT_LABEL_MAX];
long pendingIndex = SAMPLE_INDEX_ASSIGN;
long pendingTMs = -1;
long sampleIndex = 0;

void logLine(const String& s) {{
    Serial.println(s);
}}

String deviceId() {{
    uint32_t upper = (uint32_t)(ESP.getEfuseMac() >> 32);
    uint32_t lower = (uint32_t)ESP.getEfuseMac();
    char id[36];
    snprintf(id, sizeof(id), "opensmell-%08X%08X", upper, lower);
    return String(id);
}}

String infoLine() {{
    // interval_ms is not optional in practice: a host that has to assume a rate
    // rescales every count-based temporal feature by the ratio of the two.
    return "INFO," + deviceId() + "," + String(FW_VERSION) + "," +
           String(N_SENSORS) + ",interval_ms=" + String(SAMPLE_INTERVAL_MS);
}}

float readVoltage(int pin) {{
    return (analogRead(pin) / (float)4095) * 3.3f;
}}

void sendToClients(const String& line) {{
    for (int i = 0; i < MAX_CLIENTS; i++) {{
        if (clients[i] && clients[i].connected()) {{
            clients[i].println(line);
        }}
    }}
}}

void announceToAll(const String& line) {{
    logLine(line);
    sendToClients(line);
}}

// Queue a phase boundary for the next emitted sample. `index` may be
// SAMPLE_INDEX_ASSIGN, in which case the device's own counter is used.
void queueEvent(const String& label, long index, long tMs) {{
    if (label.length() == 0 || label.length() >= EVENT_LABEL_MAX) {{
        logLine("ERR,8,EVENT label must be 1.." + String(EVENT_LABEL_MAX - 1) + " characters");
        return;
    }}
    strncpy(pendingLabel, label.c_str(), EVENT_LABEL_MAX - 1);
    pendingLabel[EVENT_LABEL_MAX - 1] = '\0';
    pendingIndex = (index == SAMPLE_INDEX_ASSIGN) ? sampleIndex : index;
    pendingTMs = tMs;
    pendingEvent = true;
}}

// Emit the queued boundary, if any, immediately before the sample it annotates.
void flushPendingEvent() {{
    if (!pendingEvent) return;
    String line = "EVENT," + String(pendingLabel) + "," + String(pendingIndex);
    if (pendingTMs >= 0) {{
        line += ",t_ms=" + String(pendingTMs);
    }}
    announceToAll(line);
    pendingEvent = false;
}}

void collectBaseline() {{
    logLine("collectBaseline: sampling " + String(BASELINE_SAMPLES) + " points per channel");
    double sum[N_SENSORS] = {{0}};
    for (int s = 0; s < BASELINE_SAMPLES; s++) {{
        for (int i = 0; i < N_SENSORS; i++) {{
            sum[i] += readVoltage(SENSOR_PINS[i]);
        }}
        delay(SAMPLE_INTERVAL_MS);
    }}
    for (int i = 0; i < N_SENSORS; i++) {{
        r0[i] = (float)(sum[i] / BASELINE_SAMPLES);
    }}
    calibrated = true;
    logLine("Baseline complete; R0 stored per channel");
}}

void setupWifi() {{
    if (strlen(WIFI_SSID) > 0) {{
        logLine("Connecting to SSID " + String(WIFI_SSID));
        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASS);
        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 40) {{
            delay(500);
            attempts++;
        }}
        if (WiFi.status() == WL_CONNECTED) {{
            logLine("WiFi connected: " + WiFi.localIP().toString());
        }} else {{
            logLine("STA connect failed; switching to SoftAP");
            WiFi.mode(WIFI_AP);
            String ap = AP_SSID_PREFIX + String((uint32_t)(ESP.getEfuseMac() & 0xFFFF), HEX);
            WiFi.softAP(ap.c_str(), "osmograph");
            logLine("SoftAP started: " + ap);
        }}
    }} else {{
        WiFi.mode(WIFI_AP);
        String ap = AP_SSID_PREFIX + String((uint32_t)(ESP.getEfuseMac() & 0xFFFF), HEX);
        WiFi.softAP(ap.c_str(), "osmograph");
        logLine("SoftAP started: " + ap);
    }}
}}

void setup() {{
    Serial.begin(115200);

    analogSetWidth(12);
    analogSetAttenuation(ADC_11DB);
    for (int i = 0; i < N_SENSORS; i++) {{
        pinMode(SENSOR_PINS[i], INPUT);
    }}

    setupWifi();

    if (!MDNS.begin("osmograph")) {{
        logLine("mDNS init failed");
    }} else {{
        MDNS.addService("_osmograph", "tcp", OSM_TCP_PORT);
        logLine("mDNS: osmograph.local advertises _osmograph._tcp:" + String(OSM_TCP_PORT));
    }}

    server.begin();
    logLine("TCP server on port " + String(OSM_TCP_PORT));
    logLine("Device " + deviceId() + " fw " + String(FW_VERSION) + " channels " + String(N_SENSORS));
    // Announce the channel layout on Serial too, so the desktop auto-detects
    // the stream width the instant the board powers on (no manual rig picker).
    logLine(infoLine());

    logLine("Collecting baseline...");
    collectBaseline();
    logLine("Streaming ready");
}}

void serviceClients() {{
    for (int i = 0; i < MAX_CLIENTS; i++) {{
        if (!clients[i] || !clients[i].connected()) {{
            clients[i] = WiFiClient();
            continue;
        }}
        while (clients[i].available()) {{
            String cmd = clients[i].readStringUntil('\n');
            cmd.trim();
            if (cmd == "PING") {{
                clients[i].println("PONG");
            }} else if (cmd.startsWith("CAL")) {{
                collectBaseline();
                clients[i].println("CAL,OK");
            }} else if (cmd.startsWith("EVENT,")) {{
                // EVENT,<label>,<sample_index>[,t_ms=<MS>]. The label is quoted
                // rather than split blindly: a comma in it would otherwise shift
                // every following field, which is exactly what ERR,8 is for.
                int c1 = cmd.indexOf(',');
                int c2 = cmd.indexOf(',', c1 + 1);
                if (c1 < 0 || c2 < 0) {{
                    clients[i].println("ERR,8,EVENT needs <label> and <sample_index>");
                }} else {{
                    String label = cmd.substring(c1 + 1, c2);
                    String rest = cmd.substring(c2 + 1);
                    long index = SAMPLE_INDEX_ASSIGN;
                    long tMs = -1;
                    int tAt = rest.indexOf("t_ms=");
                    if (tAt >= 0) {{
                        tMs = strtol(rest.substring(tAt + 5).c_str(), NULL, 10);
                        rest = rest.substring(0, tAt);
                    }}
                    if (rest.length() == 0) {{
                        clients[i].println("ERR,8,EVENT needs a sample_index");
                    }} else {{
                        index = strtol(rest.c_str(), NULL, 10);
                        queueEvent(label, index, tMs);
                    }}
                }}
            }}
        }}
    }}
}}

void loop() {{
    WiFiClient newClient = server.available();
    if (newClient) {{
        for (int i = 0; i < MAX_CLIENTS; i++) {{
            if (!clients[i]) {{
                clients[i] = newClient;
                logLine("Client connected (slot " + String(i) + ")");
                // Every client gets the cadence, not just the first one: a client
                // that connects to a running stream has no other way to learn it.
                clients[i].println(infoLine());
                break;
            }}
        }}
    }}

    serviceClients();

    uint32_t now = millis();
    if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {{
        lastSampleMs = now;
        flushPendingEvent();
        String line = "OSM";
        for (int i = 0; i < N_SENSORS; i++) {{
            line += ",";
            line += String(readVoltage(SENSOR_PINS[i]), 4);
        }}
        Serial.println(line);
        sendToClients(line);
        sampleIndex++;
    }}

    yield();
}}
"#)
}

/// Generate a serial-only Arduino firmware sketch that runs on AVR boards
/// (Arduino Uno / Nano / Mega) — no WiFi, no TCP, no mDNS.
///
/// Streams `OSM` readings at 10 Hz over Serial at the same format as the ESP32
/// sketch (`OSM,<volts>,...` after a `INFO,...` boot line), so the desktop
/// auto-detects the channel width exactly like any other controller. Sensor
/// pins are AVR analog constants (`A0`…`A5`), read at 10-bit / 5 V and scaled
/// to volts so values are comparable with an ESP32 rig regardless of board.
pub fn generate_avr_sketch(sensor_pins: &[&str]) -> String {
    let pins_str = sensor_pins.iter()
        .map(|p| format!("  {}", p))
        .collect::<Vec<_>>()
        .join(",\n");
    let n_sensors = sensor_pins.len();

    format!(r#"
// OpenSmell Arduino Firmware (AVR: Uno / Nano / Mega)
// Generated by opensmell-rs protocol::generate_avr_sketch
// Serial-only; streams OSM readings at 10 Hz. Pins: {sensor_pins:?}

#define FW_VERSION "1.1.0-avr"
#define SAMPLE_INTERVAL_MS 100          // 10 Hz per channel
#define BASELINE_SECONDS 30
#define BASELINE_SAMPLES (BASELINE_SECONDS * 10)

// Sensor configuration
const int SENSOR_PINS[] = {{
{pins_str}
}};
const int N_SENSORS = {n_sensors};

// Baseline (R0) values, updated during calibration
float r0[N_SENSORS] = {{0}};
bool calibrated = false;
unsigned long lastSampleMs = 0;

void logLine(const String& s) {{
    Serial.println(s);
}}

String deviceId() {{
    // AVR has no efuse; derive a stable-ish ID from the first channel's ADC so
    // each board can be told apart in the fleet.
    uint32_t sig = (uint32_t)analogRead(SENSOR_PINS[0]);
    char id[32];
    snprintf(id, sizeof(id), "opensmell-avr-%08lX", (unsigned long)(0xA7000000UL | (sig & 0x00FFFFFFUL)));
    return String(id);
}}

float readVoltage(int pin) {{
    return (analogRead(pin) / (float)1023) * 5.0f;
}}

void collectBaseline() {{
    logLine("collectBaseline: sampling " + String(BASELINE_SAMPLES) + " points per channel");
    double sum[N_SENSORS] = {{0}};
    for (int s = 0; s < BASELINE_SAMPLES; s++) {{
        for (int i = 0; i < N_SENSORS; i++) {{
            sum[i] += readVoltage(SENSOR_PINS[i]);
        }}
        delay(SAMPLE_INTERVAL_MS);
    }}
    for (int i = 0; i < N_SENSORS; i++) {{
        r0[i] = (float)(sum[i] / BASELINE_SAMPLES);
    }}
    calibrated = true;
    logLine("Baseline complete; R0 stored per channel");
}}

void setup() {{
    Serial.begin(115200);
    for (int i = 0; i < N_SENSORS; i++) {{
        pinMode(SENSOR_PINS[i], INPUT);
    }}
    logLine("Device " + deviceId() + " fw " + String(FW_VERSION) + " channels " + String(N_SENSORS));
    logLine("INFO," + deviceId() + "," + String(FW_VERSION) + "," + String(N_SENSORS));
    logLine("Collecting baseline...");
    collectBaseline();
    logLine("Streaming ready");
}}

void loop() {{
    unsigned long now = millis();
    if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {{
        lastSampleMs = now;
        String line = "OSM";
        for (int i = 0; i < N_SENSORS; i++) {{
            line += ",";
            line += String(readVoltage(SENSOR_PINS[i]), 4);
        }}
        Serial.println(line);
    }}
}}
"#)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_osm_line() {
        let protocol = OsmProtocol::new(3);
        let msg = protocol.parse_line("OSM,1.234,2.345,3.456", 0.0).unwrap();
        match msg {
            OsmMessage::Data { channels, .. } => {
                assert_eq!(channels.len(), 3);
                assert!((channels[0] - 1.234).abs() < 0.001);
            }
            _ => panic!("Expected Data message"),
        }
    }

    #[test]
    fn test_parse_kinds_mixed_rig() {
        let protocol = OsmProtocol::new(0);
        match protocol.parse_line("OSMK,analog,mems,env_temp,env_hum,fan", 0.0).unwrap() {
            OsmMessage::Kinds { kinds } => {
                assert_eq!(kinds, vec![
                    ChannelKind::AnalogMox,
                    ChannelKind::MemsIndex,
                    ChannelKind::EnvTemp,
                    ChannelKind::EnvHum,
                    ChannelKind::Fan,
                ]);
            }
            _ => panic!("Expected Kinds message"),
        }
    }

    #[test]
    fn test_parse_kinds_lenient_unknown_tokens() {
        let protocol = OsmProtocol::new(0);
        match protocol.parse_line("OSMK,analog,bogus,mems", 0.0).unwrap() {
            OsmMessage::Kinds { kinds } => {
                assert_eq!(kinds[0], ChannelKind::AnalogMox);
                assert_eq!(kinds[1], ChannelKind::AnalogMox); // unknown → lenient default
                assert_eq!(kinds[2], ChannelKind::MemsIndex);
            }
            _ => panic!("Expected Kinds message"),
        }
    }

    #[test]
    fn test_parse_env_telemetry() {
        let protocol = OsmProtocol::new(0);
        match protocol.parse_line("ENV,24.5,52.0", 0.0).unwrap() {
            OsmMessage::Env { temperature, humidity } => {
                assert!((temperature - 24.5).abs() < 0.001);
                assert!((humidity - 52.0).abs() < 0.001);
            }
            _ => panic!("Expected Env message"),
        }
    }

    #[test]
    fn test_parse_env_too_short() {
        let protocol = OsmProtocol::new(0);
        assert!(protocol.parse_line("ENV,24.5", 0.0).is_err());
    }

    #[test]
    fn test_parse_event_phase_boundary() {
        let protocol = OsmProtocol::new(0);
        match protocol.parse_line("EVENT,exposure,600", 0.0).unwrap() {
            OsmMessage::Event { label, sample_index, t_ms } => {
                assert_eq!(label, "exposure");
                assert_eq!(sample_index, 600);
                assert_eq!(t_ms, None);
            }
            _ => panic!("Expected Event message"),
        }
    }

    #[test]
    fn test_parse_event_carries_optional_host_clock() {
        let protocol = OsmProtocol::new(0);
        match protocol.parse_line("EVENT,recovery,720,t_ms=361200", 0.0).unwrap() {
            OsmMessage::Event { label, sample_index, t_ms } => {
                assert_eq!(label, "recovery");
                assert_eq!(sample_index, 720);
                assert_eq!(t_ms, Some(361200));
            }
            _ => panic!("Expected Event message"),
        }
    }

    #[test]
    fn test_parse_event_device_assigned_index_survives() {
        let protocol = OsmProtocol::new(0);
        match protocol
            .parse_line(&format_event("baseline", SAMPLE_INDEX_DEVICE_ASSIGNED, None), 0.0)
            .unwrap()
        {
            OsmMessage::Event { label, sample_index, t_ms } => {
                assert_eq!(label, "baseline");
                assert_eq!(sample_index, SAMPLE_INDEX_DEVICE_ASSIGNED);
                assert_eq!(t_ms, None);
            }
            _ => panic!("Expected Event message"),
        }
    }

    #[test]
    fn test_parse_event_too_short_or_unlabelled_is_an_error() {
        let protocol = OsmProtocol::new(0);
        assert!(protocol.parse_line("EVENT", 0.0).is_err());
        assert!(protocol.parse_line("EVENT,exposure", 0.0).is_err());
        assert!(protocol.parse_line("EVENT,,600", 0.0).is_err());
    }

    #[test]
    fn test_event_labels_match_the_data_commons_vocabulary() {
        // Every conventional label must parse and survive a round trip, since the
        // validator treats an unknown label as worth reporting.
        for label in KNOWN_PHASE_LABELS {
            match protocol_event(label) {
                OsmMessage::Event { label: parsed, .. } => assert_eq!(parsed, label),
                _ => panic!("Expected Event message"),
            }
        }
    }

    fn protocol_event(label: &str) -> OsmMessage {
        let protocol = OsmProtocol::new(0);
        protocol
            .parse_line(&format_event(label, SAMPLE_INDEX_DEVICE_ASSIGNED, None), 0.0)
            .expect("conventional label parses")
    }

    #[test]
    fn test_format_event_shapes() {
        assert_eq!(format_event("baseline", -1, None), "EVENT,baseline,-1");
        assert_eq!(format_event("exposure", 600, None), "EVENT,exposure,600");
        assert_eq!(
            format_event("recovery", 720, Some(361200)),
            "EVENT,recovery,720,t_ms=361200"
        );
    }

    #[test]
    fn test_event_round_trips_through_the_parser() {
        let protocol = OsmProtocol::new(0);
        for (label, index, t_ms) in [
            ("baseline", 0i64, None),
            ("exposure", 600, Some(300_000i64)),
            ("gap", 720, Some(360_000)),
            ("recovery", 960, None),
        ] {
            let line = format_event(label, index, t_ms);
            match protocol.parse_line(&line, 0.0).unwrap() {
                OsmMessage::Event { label: l, sample_index: i, t_ms: t } => {
                    assert_eq!((l.as_str(), i, t), (label, index, t_ms));
                }
                _ => panic!("Expected Event message"),
            }
        }
    }

    #[test]
    fn test_unknown_message_type_is_skipped_not_fatal() {
        // The backward-compatibility contract for EVENT: an older reader meets it
        // as an unknown line and carries on.
        let protocol = OsmProtocol::new(3);
        match protocol.parse_line("EVENT,exposure,600", 0.0).unwrap() {
            OsmMessage::Event { .. } => {}
            other => panic!("this build understands EVENT, got {:?}", other),
        }
        // A type this build has never heard of is Unknown, not an error.
        match protocol.parse_line("FUTURE,1,2,3", 0.0).unwrap() {
            OsmMessage::Unknown(raw) => assert_eq!(raw, "FUTURE,1,2,3"),
            other => panic!("Expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn test_generated_sketch_has_mdns_and_protocol() {
        let sketch = generate_arduino_sketch(&[32, 33, 34, 35, 36, 39], "TestNet", "secret");
        assert!(sketch.contains("MDNS.addService(\"_osmograph\", \"tcp\", OSM_TCP_PORT)"));
        assert!(sketch.contains("MDNS.begin(\"osmograph\")"));
        assert!(sketch.contains("WiFiServer server(OSM_TCP_PORT);"));
        assert!(sketch.contains("\"TestNet\""));
        assert!(sketch.contains("INFO,\" + deviceId() + \",\" + String(FW_VERSION)"));
        assert!(sketch.contains("clients[i].println(\"PONG\")"));
        assert!(sketch.contains("SAMPLE_INTERVAL_MS 100"));
        assert!(sketch.contains("const int N_SENSORS = 6;"));
    }

    #[test]
    fn test_generated_sketch_declares_cadence_to_every_client() {
        // A client that connects to an already-running stream has no other way to
        // learn the rate, and guessing it rescales every temporal feature.
        let sketch = generate_arduino_sketch(&[32, 33], "", "");
        assert!(sketch.contains("String infoLine()"));
        assert!(sketch.contains("interval_ms=\" + String(SAMPLE_INTERVAL_MS)"));
        assert!(sketch.contains("clients[i].println(infoLine());"));
        assert!(sketch.contains("logLine(infoLine());"));
    }

    #[test]
    fn test_generated_sketch_handles_phase_events() {
        let sketch = generate_arduino_sketch(&[32, 33], "", "");
        assert!(sketch.contains("cmd.startsWith(\"EVENT,\")"));
        assert!(sketch.contains("queueEvent(label, index, tMs);"));
        assert!(sketch.contains("flushPendingEvent();"));
        // The boundary must reach the sample it annotates, so it is flushed
        // before the OSM line and not after it.
        let flush = sketch.find("flushPendingEvent();").expect("flush in loop");
        let osm = sketch.find("String line = \"OSM\";").expect("OSM line in loop");
        assert!(flush < osm);
        assert!(sketch.contains("announceToAll(line);"));
        assert!(sketch.contains("sampleIndex++;"));
        assert!(sketch.contains("ERR,8,EVENT needs <label> and <sample_index>"));
    }

    #[test]
    fn test_generated_sketch_ap_fallback_when_ssid_empty() {
        let sketch = generate_arduino_sketch(&[32, 33], "", "");
        assert!(sketch.contains("const char* WIFI_SSID = \"\";"));
        assert!(sketch.contains("WiFi.softAP(ap.c_str(), \"osmograph\")"));
    }

    #[test]
    fn test_generate_avr_sketch_uses_analog_pins_and_serial() {
        let sketch = generate_avr_sketch(&["A0", "A1", "A2"]);
        assert!(sketch.contains("const int N_SENSORS = 3;"));
        assert!(sketch.contains("Serial.begin(115200)"));
        assert!(sketch.contains("INFO,"));
        assert!(sketch.contains("readVoltage(SENSOR_PINS[i])"));
        assert!(sketch.contains("analogRead(pin) / (float)1023"));
        // Serial-only: no WiFi / mDNS / TCP
        assert!(!sketch.contains("#include <WiFi.h>"));
        assert!(!sketch.contains("WiFiServer"));
    }
}
