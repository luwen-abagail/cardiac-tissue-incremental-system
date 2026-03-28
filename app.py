from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardiac_system.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

db = SQLAlchemy(app)

# 数据库模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)

class IncrementalTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    task_name = db.Column(db.String(200), nullable=False)
    model_type = db.Column(db.String(50))  # 'existing' or 'custom'
    data_type = db.Column(db.String(50))   # 'existing' or 'custom'
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    result = db.Column(db.Text)  # 存储结果

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('uploads/models', exist_ok=True)
os.makedirs('uploads/data', exist_ok=True)

# 登录装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            return render_template('register.html', error='All fields are required'), 400
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match'), 400
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists'), 400
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already exists'), 400
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            return render_template('register.html', error='Registration failed'), 500
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password'), 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session.get('user_id')
    tasks = IncrementalTask.query.filter_by(user_id=user_id).all()
    return render_template('dashboard.html', tasks=tasks)

@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html')

@app.route('/incremental-learn')
@login_required
def incremental_learn():
    return render_template('incremental_learn.html')

@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_files():
    try:
        user_id = session.get('user_id')
        
        if 'model_file' not in request.files or 'data_file' not in request.files:
            return jsonify({'error': 'Missing files'}), 400
        
        model_file = request.files['model_file']
        data_file = request.files['data_file']
        
        if model_file.filename == '' or data_file.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        # 保存文件
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', f"{user_id}_{model_file.filename}")
        data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data', f"{user_id}_{data_file.filename}")
        
        model_file.save(model_path)
        data_file.save(data_path)
        
        return jsonify({'success': True, 'message': 'Files uploaded successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-learning', methods=['POST'])
@login_required
def start_learning():
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        task_name = data.get('task_name', 'Incremental Learning Task')
        model_type = data.get('model_type')  # 'existing' or 'custom'
        data_type = data.get('data_type')    # 'existing' or 'custom'
        
        task = IncrementalTask(
            user_id=user_id,
            task_name=task_name,
            model_type=model_type,
            data_type=data_type,
            status='processing'
        )
        
        db.session.add(task)
        db.session.commit()
        
        # 这里调用你的增量学习代码
        from incremental_learning import run_incremental_learning
        result = run_incremental_learning(user_id, task.id, model_type, data_type)
        
        task.status = 'completed'
        task.result = result
        db.session.commit()
        
        return jsonify({'success': True, 'task_id': task.id, 'result': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
