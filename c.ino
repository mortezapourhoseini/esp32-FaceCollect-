#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "camera_pins.h"

// WiFi credentials
const char *ssid = "XXXX";
const char *password = "XXXXX";

WebServer server(80);

// Dummy classifier function (replace with real ML inference)
String classifyImage(camera_fb_t *fb) {
  // Placeholder logic: return "Morteza" or "Other"
  // In real deployment, run model inference here
  // Example: send image to model running on edge/server and receive result
  if (fb->len % 2 == 0) return "Morteza";
  else return "Other";
}

void handleStream() {
  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println();

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      break;
    }

    String result = classifyImage(fb);

    // HTML overlay for label
    String html = "<div style='position:absolute; bottom:5px; left:5px; \
      background:#000; color:#fff; padding:4px; font-size:16px;'>" + result + "</div>";

    client.println("--frame");
    client.println("Content-Type: image/jpeg");
    client.println("Content-Length: " + String(fb->len));
    client.println();
    client.write(fb->buf, fb->len);
    client.println();

    // Display classification result in browser log (JS suggestion)
    client.println("<script>console.log('Prediction: " + result + "');</script>");

    esp_camera_fb_return(fb);
    delay(100);
  }
}

void handleHTML() {
  server.send(200, "text/html", R"rawliteral(
    <html>
    <head>
      <meta http-equiv='refresh' content='1'>
      <title>ESP32-CAM Stream</title>
      <style>
        body { margin: 0; background: #000; }
        img { width: 100%; }
        .label {
          position: absolute;
          bottom: 10px;
          left: 10px;
          background-color: rgba(0, 0, 0, 0.5);
          color: white;
          padding: 5px 10px;
          font-size: 18px;
          border-radius: 5px;
        }
      </style>
    </head>
    <body>
      <div style='position:relative;'>
        <img src='/stream'>
      </div>
    </body>
    </html>
  )rawliteral");
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    return;
  }

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleHTML);
  server.on("/stream", HTTP_GET, handleStream);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  delay(2);
}

