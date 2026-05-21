#include <Wire.h>
#define SDA_PIN 22
#define SCL_PIN 23
uint8_t MPU_ADDR = 0x68; 

#define REG_PWR_MGMT_1    0x6B
#define REG_ACCEL_CONFIG  0x1C
#define REG_ACCEL_XOUT_H  0x3B

const float ACC_SCALE = 16384.0f;
bool i2cWrite8(uint8_t dev, uint8_t reg, uint8_t data){
  Wire.beginTransmission(dev);
  Wire.write(reg); Wire.write(data);
  return Wire.endTransmission(true)==0;
}
bool i2cReadN(uint8_t dev, uint8_t reg, uint8_t* buf, size_t len){
  Wire.beginTransmission(dev);
  Wire.write(reg);
  if (Wire.endTransmission(false)!=0) return false;
  size_t n = Wire.requestFrom(dev,(uint8_t)len,(uint8_t)true);
  if (n!=len) return false;
  for(size_t i=0;i<len;i++) buf[i]=Wire.read();
  return true;
}
bool probe(uint8_t addr){
  Wire.beginTransmission(addr);
  return Wire.endTransmission(true)==0;
}
bool mpuInit(uint8_t addr){
  if(!i2cWrite8(addr, REG_PWR_MGMT_1, 0x00)) return false; 
  delay(100);
  if(!i2cWrite8(addr, REG_ACCEL_CONFIG, 0x00)) return false; 
  delay(10);
  return true;
}

void setup(){
  Serial.begin(921600);
  delay(300);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if(!probe(MPU_ADDR)) MPU_ADDR = 0x69;
  if(!probe(MPU_ADDR)){
    Serial.println("ERR: MPU6050 not found (0x68/0x69).");
    while(1) delay(500);
  }
  if(!mpuInit(MPU_ADDR)){
    Serial.println("ERR: MPU6050 init failed.");
    while(1) delay(500);
  }

  Serial.println("timestamp_ms,ax,ay,az");
}

void loop(){
  uint8_t raw[6];
  if (i2cReadN(MPU_ADDR, REG_ACCEL_XOUT_H, raw, 6)) {
    int16_t axr=(raw[0]<<8)|raw[1];
    int16_t ayr=(raw[2]<<8)|raw[3];
    int16_t azr=(raw[4]<<8)|raw[5];

    float ax = axr/ACC_SCALE;
    float ay = ayr/ACC_SCALE;
    float az = azr/ACC_SCALE;

    unsigned long t = millis();
    Serial.print(t); Serial.print(',');
    Serial.print(ax,6); Serial.print(',');
    Serial.print(ay,6); Serial.print(',');
    Serial.println(az,6);
  }
  delay(10);
}
