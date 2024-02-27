from flask import Flask, render_template, request

app = Flask(__name__) # App 생성 

@app.route('/') # 기본 url   
def index() :
    return render_template('index.html') # 메뉴 선택

    
@app.route('/chart')
def info():
    return render_template('chart.html')




@app.route('/service2') 
def service2(): 
    return render_template('/service2_main.html') # texts변수 




@app.route("/service2_result", methods =['POST'])
def service2_result():
    texts = request.form['texts']      
    
    # 텍스트분류 모델 import 
    from mbti_pred import classifier      
    y_pred_result = classifier(texts)  
    
    return render_template('/service2_result.html',
                           texts=texts, y_pred_result=y_pred_result)    
            

if __name__ == '__main__':
    app.run(port=80)
