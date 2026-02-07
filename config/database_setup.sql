-- ========================================
-- 卡密服务器数据库初始化脚本
-- 在 Supabase SQL Editor 中执行
-- ========================================

-- 1. 创建卡密表
CREATE TABLE IF NOT EXISTS licenses (
    id BIGSERIAL PRIMARY KEY,
    license_key VARCHAR(24) UNIQUE NOT NULL,
    machine_id VARCHAR(64),
    status VARCHAR(20) DEFAULT 'unused',
    created_at TIMESTAMP DEFAULT NOW(),
    activated_at TIMESTAMP,
    expires_at TIMESTAMP,
    max_devices INTEGER DEFAULT 1,
    notes TEXT,
    created_by VARCHAR(100),
    last_check_at TIMESTAMP
);

-- 2. 创建索引（提高查询性能）
CREATE INDEX IF NOT EXISTS idx_license_key ON licenses(license_key);
CREATE INDEX IF NOT EXISTS idx_machine_id ON licenses(machine_id);
CREATE INDEX IF NOT EXISTS idx_status ON licenses(status);
CREATE INDEX IF NOT EXISTS idx_expires_at ON licenses(expires_at);

-- 3. 创建激活日志表（记录所有激活尝试）
CREATE TABLE IF NOT EXISTS activation_logs (
    id BIGSERIAL PRIMARY KEY,
    license_key VARCHAR(24) NOT NULL,
    machine_id VARCHAR(64),
    action VARCHAR(50),
    success BOOLEAN,
    message TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_activation_license ON activation_logs(license_key);
CREATE INDEX IF NOT EXISTS idx_activation_time ON activation_logs(created_at);

-- 4. 创建设备绑定表（支持多设备激活）
CREATE TABLE IF NOT EXISTS device_bindings (
    id BIGSERIAL PRIMARY KEY,
    license_key VARCHAR(24) NOT NULL,
    machine_id VARCHAR(64) NOT NULL,
    device_name VARCHAR(100),
    activated_at TIMESTAMP DEFAULT NOW(),
    last_check_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(license_key, machine_id)
);

CREATE INDEX IF NOT EXISTS idx_device_license ON device_bindings(license_key);
CREATE INDEX IF NOT EXISTS idx_device_machine ON device_bindings(machine_id);

-- 5. 插入测试卡密（用于开发测试）
INSERT INTO licenses (license_key, status, expires_at, max_devices, notes) VALUES
('KIRO-TEST-1234-5678-ABCD', 'unused', NOW() + INTERVAL '365 days', 1, '测试卡密 - 单设备'),
('KIRO-DEMO-AAAA-BBBB-CCCC', 'unused', NOW() + INTERVAL '365 days', 3, '演示卡密 - 3设备'),
('KIRO-FREE-XXXX-YYYY-ZZZZ', 'unused', NOW() + INTERVAL '30 days', 1, '免费试用 - 30天')
ON CONFLICT (license_key) DO NOTHING;

-- 6. 创建管理密码表（用于卡密管理工具登录）
CREATE TABLE IF NOT EXISTS admin_password (
    id INTEGER PRIMARY KEY DEFAULT 1,
    password_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- 插入默认密码（hye19911206 的 SHA-256 哈希）
INSERT INTO admin_password (id, password_hash) VALUES
(1, '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918')
ON CONFLICT (id) DO UPDATE SET password_hash = EXCLUDED.password_hash;

-- 6. 创建视图：查看活跃卡密
CREATE OR REPLACE VIEW active_licenses AS
SELECT 
    l.license_key,
    l.max_devices,
    COUNT(d.machine_id) as device_count,
    l.activated_at,
    l.expires_at,
    EXTRACT(DAY FROM (l.expires_at - NOW())) as days_remaining
FROM licenses l
LEFT JOIN device_bindings d ON l.license_key = d.license_key
WHERE l.status = 'active'
  AND l.expires_at > NOW()
GROUP BY l.license_key, l.max_devices, l.activated_at, l.expires_at
ORDER BY l.activated_at DESC;

-- 7. 创建视图：统计信息
CREATE OR REPLACE VIEW license_stats AS
SELECT 
    COUNT(*) as total_licenses,
    COUNT(CASE WHEN status = 'unused' THEN 1 END) as unused_count,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
    COUNT(CASE WHEN status = 'disabled' THEN 1 END) as disabled_count,
    COUNT(CASE WHEN expires_at < NOW() THEN 1 END) as expired_count
FROM licenses;

-- 8. 创建函数：自动记录激活日志
CREATE OR REPLACE FUNCTION log_activation()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'active' AND OLD.status = 'unused' THEN
        INSERT INTO activation_logs (license_key, machine_id, action, success, message)
        VALUES (NEW.license_key, NEW.machine_id, 'activate', true, '首次激活');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 8. 创建触发器
DROP TRIGGER IF EXISTS trigger_log_activation ON licenses;
CREATE TRIGGER trigger_log_activation
    AFTER UPDATE ON licenses
    FOR EACH ROW
    EXECUTE FUNCTION log_activation();

-- 9. 查询所有卡密（验证安装）
SELECT 
    license_key,
    status,
    machine_id,
    TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at,
    TO_CHAR(expires_at, 'YYYY-MM-DD HH24:MI:SS') as expires_at,
    notes
FROM licenses
ORDER BY created_at DESC;

-- 10. 查看统计信息
SELECT * FROM license_stats;

-- ========================================
-- 常用查询语句
-- ========================================

-- 查看所有未使用的卡密
-- SELECT license_key, expires_at FROM licenses WHERE status = 'unused';

-- 查看所有活跃卡密
-- SELECT * FROM active_licenses;

-- 查看最近的激活记录
-- SELECT * FROM activation_logs ORDER BY created_at DESC LIMIT 10;

-- 禁用某个卡密
-- UPDATE licenses SET status = 'disabled' WHERE license_key = 'KIRO-XXXX-XXXX-XXXX-XXXX';

-- 解绑设备（换机）
-- UPDATE licenses SET machine_id = NULL, status = 'unused' WHERE license_key = 'KIRO-XXXX-XXXX-XXXX-XXXX';

-- 延长有效期
-- UPDATE licenses SET expires_at = expires_at + INTERVAL '365 days' WHERE license_key = 'KIRO-XXXX-XXXX-XXXX-XXXX';

-- 重置管理密码为默认密码（hye19911206）
-- UPDATE admin_password SET password_hash = '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', updated_at = NOW() WHERE id = 1;

-- 批量生成卡密（需要先生成卡密字符串）
-- INSERT INTO licenses (license_key, status, expires_at) VALUES
-- ('KIRO-XXXX-XXXX-XXXX-XXX1', 'unused', NOW() + INTERVAL '365 days'),
-- ('KIRO-XXXX-XXXX-XXXX-XXX2', 'unused', NOW() + INTERVAL '365 days'),
-- ('KIRO-XXXX-XXXX-XXXX-XXX3', 'unused', NOW() + INTERVAL '365 days');
