

from tkinter import *
from tkinter import ttk

import numpy as np
import pandas as pd
from tkinter import filedialog

# ML agos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class start():
    def __init__(self):
        print(" started")

    def start_fn(self):
        ob = home_page()
        ob.home_fn()


class home_page(start):
    global data, path

    def home_fn(self):
        global x_index, y_index, selected_x, selected_y
        x_index = []
        y_index = []
        selected_x = []
        selected_y = []

        def select_x_fn():
            x = select_col.get()
            selected_x.append(x)
            ind_x = data.columns.get_loc(x)
            x_index.append(ind_x)
            x_txt_box.insert(END, x + '\n')

        def select_y_fn():
            y = select_col.get()
            selected_y.append(y)
            ind_y = data.columns.get_loc(y)
            y_index.append(ind_y)
            y_txt_box.insert(END, y + '\n')

        def page_2_fn():
            x_col = x_index
            y_col = y_index
            name_x = selected_x
            name_y = selected_y
            win.destroy()
            obj = algorithm_choose()
            obj.algorithms(path, x_col, y_col, name_x, name_y)

        def open_file_fn():
            global path, data
            file = filedialog.askopenfilename(title="Open File", filetypes=(
            ("csv files", "*.csv"), ("Text Files", "*.txt"), ("All files", "*."), ("Python files", "*.py")))
            path = file.replace("/", "//")
            data = pd.read_csv(path)
            list_1 = list(data.columns)
            select_col['values'] = tuple(list_1)

        def refresh_fn():
            x_txt_box.delete('1.0', END)
            y_txt_box.delete('1.0', END)
            y_index.clear()
            x_index.clear()
            selected_x.clear()
            selected_y.clear()

        win = Tk()
        win.geometry('1250x1000')
        win.title("Prediction Algorithms")

        frame1 = Frame(win, bg='#8678ab')
        frame1.pack(ipadx=50, ipady=50, expand=True, fill='both')

        header_l = Label(frame1, text='PREDICTIONS', font=("Pacifica", "28", "bold"), bg='#e9692c', relief=RAISED, padx=10)
        header_l.pack(ipady=20)

        select_col_en = StringVar()
        select_col = ttk.Combobox(frame1, width=50, textvariable=select_col_en)
        select_col.place(x=420, y=200)

        select_x_btn = Button(frame1, text='Independent X', font=("Pacifica","16","bold"),bg="orange", relief=GROOVE, command=select_x_fn)
        select_x_btn.place(x=300, y=300)

        select_y_btn = Button(frame1, text='Dependent Y',font=("Pacifica","16","bold"),bg="orange", relief=GROOVE, command=select_y_fn)
        select_y_btn.place(x=780, y=300)

        x_txt_box = Text(frame1, font=("Pacifica", '14', "bold"), width=35, height=10)
        x_txt_box.place(x=120, y=390)

        y_txt_box = Text(frame1, font=("Pacifica", '14', "bold"), width=35, height=10)
        y_txt_box.place(x=650, y=390)

        next_btn = Button(frame1, text='Next',font=("Pacifica","20","bold"),bg="#16c119", relief=GROOVE, command=page_2_fn)
        next_btn.place(x=520, y=420)

        refresh_btn = Button(frame1, text='Refresh',font=("Pacifica","20","bold"),bg="#8f2c0c", relief=GROOVE, command=refresh_fn)
        refresh_btn.place(x=520, y=500)

        open_btn = Button(frame1, text='Browse',font=("Pacifica","20","bold"),bg="#0c8f5a", relief=GROOVE, command=open_file_fn)
        open_btn.place(x=550, y=130)

        win.mainloop()


class algorithm_choose(home_page):

    def algorithms(self, path, x_col, y_col, name_x, name_y):
        win1 = Tk()
        win1.geometry('1250x1000')
        win1.title("Prediction Algorithms")
        win1.config(background='#8678ab')
        frame2 = Frame(win1, bg='#8678ab')
        frame2.pack(ipadx=50, ipady=50, expand=True, fill='both')

        def slin_reg_fn():  # Simple linear regression

            frame2.destroy()

            header_l = Label(win1, text='Simple-Linear Regression', font=("Pacifica", "28", "bold"), bg='#e9692c',
                             relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame5 = Frame(win1, bg='#8678ab')
            frame5.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_slin_reg_fn():  # Simple linear regression model
                entry = col_1_e.get()
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
                regressor = LinearRegression()
                regressor.fit(x_train, y_train)
                y_pred = regressor.predict(x_test)
                pred_y = regressor.predict([[entry]])
                print(pred_y)
                pred_l = Label(frame5, text=pred_y[0][0].round(2), font=("Pacifica", "18", "bold"), bg='#add8e6')
                pred_l.place(x=620, y=250)

            col_1_l = Label(frame5, text=f'{name_x[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_1_l.place(x=370, y=150)

            col_1_en = StringVar()
            col_1_e = Entry(frame5, width=20, textvariable=col_1_en)
            col_1_e.place(x=620, y=155)

            result_l = Label(frame5, text=f"{name_y[0]}", font=("Pacifica", "18", "bold"), bg='#ffa07a')
            result_l.place(x=370, y=250)

            result_btn = Button(frame5, text='Click to Predict', bg='#98fb98', relief=GROOVE, command=pred_slin_reg_fn)
            result_btn.place(x=510, y=355)

        def mlin_reg_fn():  # Multi linear regression
            frame2.destroy()
            header_l = Label(win1, text='Multi-Linear Regression', font=("Pacifica", "28", "bold"), bg='#e9692c',
                             relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame6 = Frame(win1, bg='#8678ab')
            frame6.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_mlin_reg_fn():  # Multi linear regression Model
                entry1 = col_1_e.get()
                entry2 = col_2_e.get()
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
                regressor = LinearRegression()
                regressor.fit(x_train, y_train)
                y_pred = regressor.predict(x_test)
                pred_y = regressor.predict([[entry1, entry2]])
                print(pred_y)
                pred_l = Label(frame6, text=pred_y[0][0].round(2), font=("Pacifica", "18", "bold"), bg='#add8e6')
                pred_l.place(x=600, y=300)

            col_1_l = Label(frame6, text=f'{name_x[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_1_l.place(x=350, y=150)

            col_2_l = Label(frame6, text=f'{name_x[1]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_2_l.place(x=350, y=210)

            col_3_l = Label(frame6, text=f'{name_y[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_3_l.place(x=350, y=300)

            col_1_en = StringVar()
            col_1_e = Entry(frame6, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            col_2_en = StringVar()
            col_2_e = Entry(frame6, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

            result_btn = Button(frame6, text='Click to Predict', bg='#98fb98', relief=GROOVE, command=pred_mlin_reg_fn)
            result_btn.place(x=460, y=375)

        def plin_reg_fn():
            frame2.destroy()
            header_l = Label(win1, text='Polynomial Regression', font=("Pacifica", "28", "bold"), bg='#e9692c', relief=RAISED,
                             padx=6)
            header_l.pack(ipady=10)

            frame7 = Frame(win1, bg='#8678ab')
            frame7.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_poly_reg_fn():
                x_entry = col_1_e.get()
                poly_reg = PolynomialFeatures(degree=7)
                x1 = poly_reg.fit_transform(x)
                regressor = LinearRegression()
                regressor.fit(x1, y)
                pred_y = regressor.predict(poly_reg.fit_transform([[x_entry]]))
                print(pred_y)
                pred_l = Label(frame7, text=pred_y[0][0].round(2), font=("Pacifica", "18", "bold"), bg='#add8e6')
                pred_l.place(x=600, y=250)

            col_1_l = Label(frame7, text=f'{name_x[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_1_l.place(x=350, y=150)

            col_1_en = StringVar()
            col_1_e = Entry(frame7, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            result_l = Label(frame7, text=f"{name_y[0]}", font=("Pacifica", "18", "bold"), bg='#ffa07a')
            result_l.place(x=350, y=250)

            result_btn = Button(frame7, text='Click to Predict', bg='#98fb98', relief=GROOVE, command=pred_poly_reg_fn)
            result_btn.place(x=460, y=355)

        # Classification------------------

        def lcls_fn():
            frame2.destroy()
            header_l = Label(win1, text='Logistic Regression', font=("Pacifica", "28", "bold"), bg='#e9692c', relief=RAISED,
                             padx=6)
            header_l.pack(ipady=10)

            frame8 = Frame(win1, bg='#8678ab')
            frame8.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_logcls_fn():
                x_entry1 = col_1_e.get()
                x_entry2 = col_2_e.get()
                sc = StandardScaler()
                x1 = sc.fit_transform(x)
                x2 = sc.fit_transform([[x_entry1, x_entry2]])
                classifier = LogisticRegression()
                classifier.fit(x1, np.ravel(y))
                pred_y = classifier.predict(x2)
                pred_l = Label(frame8, text=pred_y[0], font=("Pacifica", "18", "bold"), bg='#add8e6')
                pred_l.place(x=600, y=300)

            col_1_l = Label(frame8, text=f'{name_x[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_1_l.place(x=350, y=150)

            col_2_l = Label(frame8, text=f'{name_x[1]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_2_l.place(x=350, y=210)

            col_3_l = Label(frame8, text=f'{name_y[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_3_l.place(x=350, y=300)

            col_1_en = StringVar()
            col_1_e = Entry(frame8, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            col_2_en = StringVar()
            col_2_e = Entry(frame8, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

            result_btn = Button(frame8, text='Click to Predict', bg='#98fb98', relief=GROOVE, command=pred_logcls_fn)
            result_btn.place(x=460, y=375)

        def naive_fn():
            frame2.destroy()
            header_l = Label(win1, text='Naive Bayes', font=("Pacifica", "28", "bold"), bg='#e9692c', relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame9 = Frame(win1, bg='#8678ab')
            frame9.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_nvbys_fn():
                x_entry1 = col_1_e.get()
                x_entry2 = col_2_e.get()
                classifier = GaussianNB()
                classifier.fit(x, y)
                pred_y = classifier.predict([[x_entry1, x_entry2]])
                print(pred_y)
                pred_l = Label(frame9, text=pred_y[0], font=("Pacifica", "18", "bold"), bg='#add8e6')
                pred_l.place(x=600, y=300)

            col_1_l = Label(frame9, text=f'{name_x[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_1_l.place(x=350, y=150)

            col_2_l = Label(frame9, text=f'{name_x[1]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_2_l.place(x=350, y=210)

            col_3_l = Label(frame9, text=f'{name_y[0]}', font=("Pacifica", "18", "bold"), bg='#ffa07a')
            col_3_l.place(x=350, y=300)

            col_1_en = StringVar()
            col_1_e = Entry(frame9, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            col_2_en = StringVar()
            col_2_e = Entry(frame9, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

            result_btn = Button(frame9, text='Click to Predict', bg='#98fb98', relief=GROOVE, command=pred_nvbys_fn)
            result_btn.place(x=460, y=375)

        dataset = pd.read_csv(path)
        x = dataset.iloc[:, x_col].values
        y = dataset.iloc[:, y_col].values

        header_l = Label(frame2, text='PREDICTION', font=("Pacifica", "28", "bold"), bg='#1f8d96', relief=RAISED, padx=6)
        header_l.pack(ipady=20)

        frame3 = Frame(frame2, bg='#8678ab')
        frame3.pack(fill=BOTH, side=LEFT, expand=True, padx=10, pady=10)

        frame4 = Frame(frame2, bg='#8678ab')
        frame4.pack(fill=BOTH, side=RIGHT, expand=True, padx=10, pady=10)

        slin_reg_btn = Button(frame3, text='Simple Linear Regression', font=("Pacifica", "16", "bold"), bg="#ff9999",
                              relief=GROOVE, command=slin_reg_fn)
        slin_reg_btn.pack(pady=20)

        mlin_reg_btn = Button(frame3, text='Multi Linear Regression', font=("Pacifica", "16", "bold"), bg="#99ff99",
                              relief=GROOVE, command=mlin_reg_fn)
        mlin_reg_btn.pack(pady=20)

        plin_reg_btn = Button(frame3, text='Polynomial Regression', font=("Pacifica", "16", "bold"), bg="#9999ff",
                              relief=GROOVE, command=plin_reg_fn)
        plin_reg_btn.pack(pady=20)

        lcls_btn = Button(frame4, text='Logistic Regression', font=("Pacifica", "16", "bold"), bg="#ff99ff",
                          relief=GROOVE, command=lcls_fn)
        lcls_btn.pack(pady=20)

        naive_btn = Button(frame4, text='Naive Bayes', font=("Pacifica", "16", "bold"), bg="#ffcc99", relief=GROOVE,
                           command=naive_fn)
        naive_btn.pack(pady=20)

        win1.mainloop()


if __name__ == "__main__":
    ob = start()
    ob.start_fn()
