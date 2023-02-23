import PySimpleGUIWeb as sg
import pandas as pd
import dash
sg.theme('DarkAmber')
layout = [ [sg.Text('some text on Row1')]
           [sg.Text('Enter something on Row 2')]
           [sg.Butter('ok'),sg.Button('canael')]
          ]
window = sg.Window('window Title', layout)
while True:
    event, values = window.read()
    if event == sg.win_closed or event == 'cancel':
        break
        print('you entered',valu[0])