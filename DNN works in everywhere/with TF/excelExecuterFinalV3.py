#Functions we had done!
#automatic execute data into month & day
#automatic execute data into final diff, final direction, trend, kinetic power, kinetic
#automatic execute data with Final Trend Kinetic
#automatic execute data for the rest as file 1 did
#take out the wrong numbers
#take only the cols that we can use

#need:
#delete the error ones
#take only the rows that we can use
#make it windowszation
#put in the coding in, like print(self.predict(xx))


import pandas as pd
import xlrd
import xlwt
import xlsxwriter


class ee:


    # all the things we may need to mention later, we just put it here roughly all together.
    def __init__(self,inputFileName,sheetname):
        self.inputFile = "inputFileName+".xlsx" # change here if you want to input your own data.
        self.outputFile = "inputFileName+" copy.xlsx" # change here if you want to change the save path.
        self.month = list()
        self.day = list()
        self.finalDiff = list()
        self.finalDirection = list()
        self.trend = list()
        self.kineticPower = list()
        self.kinetic = list()
        self.sheetname = sheetname


    # data scanner
    def ds(self):
        #short for dataSource

        df = pd.read_excel(self.inputFile, sheet_name=self.sheetname)     # add sheetname in so we can put all data in one excel
        j = list()
        i = 0
        while (i < len(df)) :
            j.append(df.iloc[i].tolist())
            i = i + 1
        return j


    # data reader
    def read(self, inputRow, inputCols):
        
        df = xlrd.open_workbook(self.inputFile)
        sheet_names= df.sheet_names()
        df1 = df.sheet_by_name("Sheet1")
        rows = df1.row_values(inputRow) # 获取第X+1行内容
        cols = df1.col_values(inputCols) # 获取第X+1列内容
        #print(rows)
        #print(cols)
        return rows, cols


    #Make the whole function work together.
    def finalFX(self,inputRow, inputCols): #for writekinetic

#-------------------------------------------------------------------Sheet 1--------------------------------------------------------------------------------------------

        #Get the original rows & cols we need
        #Row down here doesn't what we need, so we don't have to pay attention to it's value.
        rows, cols = self.read(inputRow, inputCols)
        rows1, cols1 = self.read(inputRow, inputCols+1)
        rows2, cols2 = self.read(inputRow, inputCols+4)
        rows3, cols3 = self.read(inputRow, inputCols+2)
        rows4, cols4 = self.read(inputRow, inputCols+3)
        rows5, cols5 = self.read(inputRow, inputCols+9)
        rows5, cols6 = self.read(inputRow, inputCols+10)
        rows5, cols7 = self.read(inputRow, inputCols+11)
        rows5, cols8 = self.read(inputRow, inputCols+12)
        rows5, cols9 = self.read(inputRow, inputCols+13)
        rows5, cols10 = self.read(inputRow, inputCols+14)
        rows5, cols11 = self.read(inputRow, inputCols+15)
        rows5, cols12 = self.read(inputRow, inputCols+16)
        

        i = 0   # write in from row i = 0
        
        #decide where you want it to be put out there
        wbk = xlsxwriter.Workbook(self.outputFile)  
        sheet = wbk.add_worksheet('Sheet1')  # we can make more sheets
        
        
        #start with the title we need down here:
        sheet.write(i,0,cols[i]) #第一列的第一行开始写入内容  
        sheet.write(i,1,"MONTH")
        sheet.write(i,2,"DAY")
        sheet.write(i,3,cols1[i])  
        sheet.write(i,4,cols2[i])  
        sheet.write(i,5,"finalDiff")
        sheet.write(i,6,"Final")
        sheet.write(i,7,"Final1")
        sheet.write(i,8,"Final2")
        sheet.write(i,9,"Trend")
        sheet.write(i,10,"Trend1")
        sheet.write(i,11,"Trend2")
        sheet.write(i,12,"TOP")  
        sheet.write(i,13,"Bottom")  
        sheet.write(i,14,"KineticPower")
        sheet.write(i,15,"Kinetic")
        sheet.write(i,16,"Kinetic1")
        sheet.write(i,17,"Kinetic2")
        sheet.write(i,18,"MVA 5")
        sheet.write(i,19,"MVA 13")
        sheet.write(i,20,"MVA 40")  
        sheet.write(i,21,"MVA 60")  
        sheet.write(i,22,"MVA 200")
        sheet.write(i,23,"MACD")
        sheet.write(i,24,"MACD SIG")
        sheet.write(i,25,"MACD HIS")


        sheet.write(i,26,"MONTH")
        sheet.write(i,27,"DAY")
        sheet.write(i,28,"OPEN")  
        sheet.write(i,29,"CLOSE")  
        sheet.write(i,30,"finalDiff")
        sheet.write(i,31,"Final")
        sheet.write(i,32,"Final up")
        sheet.write(i,33,"Final down")
        sheet.write(i,34,"Trend")
        sheet.write(i,35,"Trend up")
        sheet.write(i,36,"Trend down")
        sheet.write(i,37,"TOP")
        sheet.write(i,38,"Bottom")  
        sheet.write(i,39,"KineticPower")
        sheet.write(i,40,"Kinetic")
        sheet.write(i,41,"Kinetic1")
        sheet.write(i,42,"Kinetic2")
        sheet.write(i,43,"MVA 5")
        sheet.write(i,44,"MVA 13")
        sheet.write(i,45,"MVA 40")  
        sheet.write(i,46,"MVA 60")  
        sheet.write(i,47,"MVA 200")
        sheet.write(i,48,"MACD")
        sheet.write(i,49,"MACD SIG")
        sheet.write(i,50,"MACD HIS")
        
        #Well we finished title, so we need to put functions in
        i = 1   # write in from row i = 1, here we need to write values in
        
        while i < len(cols):
            s = str(i+1) #make it str and it stands for the row number, otherwise below formula function can't take ints
            s1 = str(i+2)

            finalDiff = "=ROUND((E"+s+"-D"+s+")*1000,2)"
            Final = "=IF(F"+s+">0,IF(F"+s+">1,2,1),IF(F"+s+"<-1,0,1))"
            trend = "=IF(F"+s+">1,IF(F"+s+"*F"+s1+">0,2,1),IF((F"+s+"-F"+s1+")<-1,0,1))"
            kineticPower = "=IF(M"+s+"=N"+s+",0,IF(F"+s+"=0,1,(F"+s+"/ABS(F"+s+")))*ROUND((M"+s+"-N"+s+")*1000,2))"  #we find some data the K is 0, so we need to consider 0 situation
            kinetic = "=IF(F"+s+">0,IF(O"+s+">5,2,1),IF(O"+s+"<-5,0,1))"  
            
            self.finalDiff.append(finalDiff)
            self.finalDirection.append(Final)
            self.trend.append(trend)

            sheet.write(i,0,cols[i]) #第一列的第一行开始写入内容
            sheet.write_formula(i,1,"=MONTH(A"+s+")")
            sheet.write_formula(i,2,"=DAY(A"+s+")")
            
            sheet.write(i,3,cols1[i])  #第0行第4列开始写入内容
            sheet.write(i,4,cols2[i])  #第0行第5列开始写入内容
            sheet.write_formula(i,5,finalDiff)
            sheet.write_formula(i,6,Final)
            sheet.write_formula(i,7,"=if(G"+s+"=2,1,0)")
            sheet.write_formula(i,8,"=if(G"+s+"=0,1,0)")
            
            sheet.write_formula(i,9,trend)
            sheet.write_formula(i,10,"=if(J"+s+"=2,1,0)")
            sheet.write_formula(i,11,"=if(J"+s+"=0,1,0)")

            sheet.write(i,12,cols3[i])  #第0行第13列开始写入内容
            sheet.write(i,13,cols4[i])  #第0行第14列开始写入内容
            sheet.write_formula(i,14,kineticPower)
            sheet.write_formula(i,15,kinetic)
            sheet.write_formula(i,16,"=if(P"+s+"=2,1,0)")
            sheet.write_formula(i,17,"=if(P"+s+"=0,1,0)")
            sheet.write(i,18,cols5[i])
            sheet.write(i,19,cols6[i])
            sheet.write(i,20,cols7[i])
            sheet.write(i,21,cols8[i])
            sheet.write(i,22,cols9[i])
            sheet.write(i,23,cols10[i])
            sheet.write(i,24,cols11[i])
            sheet.write(i,25,cols12[i])
            i += 1
            #We have finished put functions in up here

            
        list = [31,34,40] # there were a few lines we don't want to write with functions directly, and here is which lines they are. you can check the line number
                          # top there when setting up the sheet

        #we will need to do norm looply, so we put all the col we need here
        colList = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  
        colList_function = ['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ']

        #Normalization func
        
        a = 26   #a stands for the line of the function starts from
        b = 1    # col line
        j = 1    # row line 
        while b < 26:  # total how many columns we need to write
            while j < len(cols):
                s = str(j+1)  # the 1st row was title which we don't want to norm directly, so we will need to do norm from the 2nd row, which means s = j(1) + 1 = 2
                
                if a in list:
                    sheet.write_formula(j,a,"="+colList[b]+s)
                else:
                    none = '""'
                    normalization = "=IF("+colList[b]+s+"="+none+","+none+",("+colList[b]+s+"-MIN("+colList[b]+":"+colList[b]+"))/(MAX("+colList[b]+":"+colList[b]+")-MIN("+colList[b]+":"+colList[b]+")))"
                    sheet.write_formula(j,a,normalization)
                j += 1
            b += 1
            a += 1
            j = 1       #remember that we need to reset j to 1 to make the inner while loop go

#------------------------------------------------------------------------------Sheet 2---------------------------------------------------------------------------------
        #Till now all data we want is already here, we need to pick the useful ones out now
        sheet1 = wbk.add_worksheet('Sheet2')

        #write in the data, actually we only need one formula

        #we need to reset i to 1, it equals to len(cols)
        i = 0

        #Write in the columns we need
        while i < len(cols):

            #cause there is no row 0 in excel so we will start from 1
            s = str(i+1)
            
            #get title
            if i==0:
                sheet1.write_formula(i,0,"=Sheet1!AG"+s)
                sheet1.write_formula(i,1,"=Sheet1!AH"+s)
                sheet1.write_formula(i,2,"=Sheet1!AJ"+s)
                sheet1.write_formula(i,3,"=Sheet1!AK"+s)
                sheet1.write_formula(i,4,"=Sheet1!AP"+s)
                sheet1.write_formula(i,5,"=Sheet1!AQ"+s)
                sheet1.write_formula(i,6,"=Sheet1!AE"+s)
                sheet1.write_formula(i,7,"=Sheet1!AN"+s)
                sheet1.write_formula(i,8,"=Sheet1!AR"+s)
                sheet1.write_formula(i,9,"=Sheet1!AS"+s)
                sheet1.write_formula(i,10,"=Sheet1!AT"+s)
                sheet1.write_formula(i,11,"=Sheet1!AU"+s)
                sheet1.write_formula(i,12,"=Sheet1!AV"+s)
                sheet1.write_formula(i,13,"=Sheet1!AW"+s)
                sheet1.write_formula(i,14,"=Sheet1!AX"+s)
                sheet1.write_formula(i,15,"=Sheet1!AY"+s)
                i += 1
                
            else:
        
                sheet1.write_formula(i,0,"=round(Sheet1!AG"+s+",4)")
                sheet1.write_formula(i,1,"=round(Sheet1!AH"+s+",4)")
                sheet1.write_formula(i,2,"=round(Sheet1!AJ"+s+",4)")
                sheet1.write_formula(i,3,"=round(Sheet1!AK"+s+",4)")
                sheet1.write_formula(i,4,"=round(Sheet1!AP"+s+",4)")
                sheet1.write_formula(i,5,"=round(Sheet1!AQ"+s+",4)")
                sheet1.write_formula(i,6,"=round(Sheet1!AE"+s+",4)")
                sheet1.write_formula(i,7,"=round(Sheet1!AN"+s+",4)")
                sheet1.write_formula(i,8,"=round(Sheet1!AR"+s+",4)")
                sheet1.write_formula(i,9,"=round(Sheet1!AS"+s+",4)")
                sheet1.write_formula(i,10,"=round(Sheet1!AT"+s+",4)")
                sheet1.write_formula(i,11,"=round(Sheet1!AU"+s+",4)")
                sheet1.write_formula(i,12,"=round(Sheet1!AV"+s+",4)")
                sheet1.write_formula(i,13,"=round(Sheet1!AW"+s+",4)")
                sheet1.write_formula(i,14,"=round(Sheet1!AX"+s+",4)")
                sheet1.write_formula(i,15,"=round(Sheet1!AY"+s+",4)")
                i += 1

            #We have finished build up sheet2

        wbk.close()

#------------------------------------------------------------------------------file 2---------------------------------------------------------------------------------

    def sd(self,inputs,j):  #short for save data, data type should be list type
        #j stands for which column it will go
        #open a new file
        self.outputFile2 = "G:/Coding mind/Machine learning/FX/"+str(j)+" weights.xlsx"
        wbk1 = xlsxwriter.Workbook(self.outputFile2)  
        sheet = wbk1.add_worksheet('Sheet1')

        #//preview process data, split them by \n #this process should be done before data input

        #//store the rows one by one into a list

        #output the list into the excel

        i = 0
        while i < len(inputs):
            s = i+1
            sheet.write(s,0,inputs[i])
            i += 1
        wbk1.close()

        


#Get any files you downloaded here
if __name__ == '__main__':
   obj = ee("EUR D")
   obj.finalFX(0,0)



