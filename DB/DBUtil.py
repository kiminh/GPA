# encoding: utf-8
# @Time    : 2019/7/15 16:41
# @Author  : zxl
# @FileName: DBUtil.py
import pymssql

class DB():
    def __init__(self,host="192.168.150.250",user="dsjfx",password="i8f66126",db="DSJFX"):
        self.host=host
        self.user=user
        self.password=password
        self.db=db
        self.conn=pymssql.connect(host=self.host,user=self.user,password=self.password,database=self.db,charset='utf8')

    def getCon(self):
        # self.conn=pymssql.connect(host=self.host,user=self.user,password=self.password,database=self.db,charset='utf8')
        cur=self.conn.cursor()
        if not cur:
            raise(NameError,"连接数据库失败")
        else:
            return cur
    def ExecQuery(self,sql):
        # cur=self.getCon()
        cur=self.conn.cursor()
        cur.execute(sql)
        res=cur.fetchall()
        #关闭连接
        # self.conn.close()
        return res

    def ExecNonQuery(self,sql):
        # cur=self.getCon()
        cur=self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()
        #关闭连接
        # self.conn.close()

    def getStuByEnrollTime(self,year):
        sql = "SELECT DISTINCT 学工号 FROM V_CUSTOMERS WHERE 院系 LIKE \'%本科生%" + str(year) + "%\'"
        res = []
        records = self.ExecQuery(sql)
        for r in records:
            res.append(r[0])
        return res
    def getDepartmentByStuid(self,stu_id):
        sql="SELECT 院系 FROM DSJFX.V_CUSTOMERS WHERE 学工号=\'"+str(stu_id)+"\'"
        res=self.ExecQuery(sql)
        return res[0][0]

    def getProfileById(self,stu_id):
        """
        性别，年级，院系
        :param stu_id:
        :return:
        """
        sql="SELECT 性别, 入学时间,出生年月,院系 FROM V_CUSTOMERS WHERE 学工号=\'"+str(stu_id)+"\' "
        res=self.ExecQuery(sql)
        return res[0]

    def getGrandByIdAndTime(self,stu_id,start_time,end_time):
        sql="SELECT 获奖名称,奖金 FROM JiangZhu WHERE 学工号 =\'"+str(stu_id)+"\' AND 时间 >= \'"+start_time+"\' AND 时间 <= \'"+end_time+"\'"
        res=[]
        records=self.ExecQuery(sql)
        for  r in records:
            grand_name=r[0]
            grand_acc=r[1]
            res.append([grand_name,grand_acc])
        return res
    def getConsumeRec(self,stu_id,start_time,end_time):
        sql = "SELECT 时间,地点,消费金额,余额  FROM V_REC_CUST_ACC " \
              "WHERE 学工号=\'"+stu_id+"\' AND 时间>\'"+start_time+"\'  AND 时间<\'"+end_time+"\'" \
              "ORDER BY 时间"
        res = []
        #print(sql)
        records = self.ExecQuery(sql)
        for r in records:
            place = r[0]
            time = r[1]
            amount=r[2]
            balence=r[3]
            res.append([place,time,amount,balence])
        return res





    "Grade这张表"
    def InsertIntoGrade(self,institute,department,enroll_time,code1,stu_id,name,
                        course,credit,grade,level,gpa,code2,ctime):
        sql="INSERT INTO Grade(institute,department,enroll_time,code1,stu_id,name,course," \
            "credit,grade,level,gpa,code2,ctime ) VALUES"+"(\'"+str(institute)+"\',\'"+str(department)+"\',\'"+str(enroll_time)+"\'," \
               "\'"+str(code1)+"\',\'"+str(stu_id)+"\',\'"+str(name)+"\',\'"+str(course)+"\',\'"+str(credit)+"\',\'"+str(grade)+"\',\'"+str(level)+"\'," \
               "\'"+str(gpa)+"\',\'"+str(code2)+"\',\'"+str(ctime)+"\')"
        self.ExecNonQuery(sql)

    def getGradeInfo(self,stu_id,semester):
        sql="SELECT course, credit,grade,level,gpa FROM Grade WHERE stu_id =\'"+stu_id+"\' AND ctime =\'"+semester+"\'"
        records=self.ExecQuery(sql)
        res=[]
        for r in records:
            course=r[0]
            credit=r[1]
            grade=r[2]
            level=r[3]
            gpa=r[4]
            res.append([course,credit,grade,level,gpa])
        return res
    def getAllGradeInfo(self,stu_id):
        sql="SELECT course, credit,grade,level,gpa,ctime FROM Grade WHERE stu_id =\'"+stu_id+"\' "
        records=self.ExecQuery(sql)
        res=[]
        for r in records:
            course=r[0]
            credit=r[1]
            grade=r[2]
            level=r[3]
            gpa=r[4]
            c_time=r[5]
            res.append([course,credit,grade,level,gpa,c_time])
        return res
    """
    Libray这张表
    """
    def InsertLibRec(self,stu_id,time,campus):
        sql="INSERT INTO Library(stu_id,time,campus) VALUES (\'"+stu_id+"\',\'"+time+"\',\'"+campus+"\')"
        self.ExecNonQuery(sql)

    def getLibInfo(self,stu_id,start_time,end_time):
        sql="SELECT time FROM Library WHERE stu_id =\'"+stu_id+"\' AND time >=\'"+start_time+"\' AND time<=\'"+end_time+"\'" \
             "ORDER BY time"
        records=self.ExecQuery(sql)
        res=[]
        for rec in records:
            t=rec[0]
            res.append(t)
        return res
    """
    Course这张表
    """
    def InsertCourse(self,dep_id,dep_name,c_id,c_name,stu_num,grade,type,teacher,week,week_num,section,classroom,campus):
        sql = "INSERT INTO Course(dep_id,dep_name,c_id,c_name,stu_num,grade,type,teacher,week,week_num,section,classroom,campus) " \
              "VALUES (\'" + str(dep_id) + "\',\'" + str(dep_name) + "\',\'" + str(c_id) + "\',\'" +str(c_name)+"\',\'" +str(stu_num)+"\',\'" +str(grade)+"\',\'" +str(type)+"\',\'" +str(teacher)+"\',\'" +str(week)+"\',\'"+str(week_num)+"\',\'" +str(section)+"\',\'" +str(classroom)+"\',\'" +str(campus)+"\')"
        self.ExecNonQuery(sql)

    def SelectCourseInfo(self,enroll_time,c_name):
        sql="SELECT dep_name,type, stu_num,teacher,id FROM Course WHERE c_name=\'"+c_name+"\' AND grade LIKE \'%"+enroll_time+"%\'"
        records=self.ExecQuery(sql)
        res=[]
        for rec in records:
            res.append([rec[0],rec[1],rec[2],rec[3],rec[4]])
        return res

if __name__ == "__main__":
    db=DB()
    res=db.SelectCourseInfo("2015","软件工程导论")
    for r in res:
        print(r)