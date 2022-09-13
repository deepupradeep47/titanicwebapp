{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "940f993d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sklearn\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import matplotlib \n",
    "from matplotlib import pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7f81147",
   "metadata": {},
   "source": [
    "# Loading the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "834031c7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=pd.read_csv('titanic.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1625d76f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a04a64de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d26aca7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Age'].fillna(data['Age'].mean(),inplace=True)#inplace =true will make parmanent "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9179b0d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Age'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "22273d55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age              0\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "23711241",
   "metadata": {},
   "outputs": [],
   "source": [
    "#lets drop cabin column\n",
    "data.drop(columns=['Cabin'],inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "845c6239",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          891 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(4)\n",
      "memory usage: 76.7+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d65476c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>61</th>\n",
       "      <td>62</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Icard, Miss. Amelie</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>113572</td>\n",
       "      <td>80.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>829</th>\n",
       "      <td>830</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>\n",
       "      <td>female</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>113572</td>\n",
       "      <td>80.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                                       Name  \\\n",
       "61            62         1       1                        Icard, Miss. Amelie   \n",
       "829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)   \n",
       "\n",
       "        Sex   Age  SibSp  Parch  Ticket  Fare Embarked  \n",
       "61   female  38.0      0      0  113572  80.0      NaN  \n",
       "829  female  62.0      0      0  113572  80.0      NaN  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#we are having 2 missing values in embarked we can get it\n",
    "data.loc[data['Embarked'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e41b5f77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "S    644\n",
       "C    168\n",
       "Q     77\n",
       "Name: Embarked, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#as majority as traveling from 's'\n",
    "data['Embarked'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f52b29cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Embarked'].fillna(\"S\",inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5ed41dd0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Embarked', ylabel='Age'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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5513VF5eruOPPz74mnnz5unRRx/VE088oTfeeEN9+/ZVXl6efD6fweSxw+FwqKCgoM25Sy+9tN2SA3SnLVu2aPv27SFj27ZtC/klCEBsMHr369tvv11VVVXtrsGwbVtpaWkqKSnRrbfeKklqbm6Wy+XSsmXLdNlll7X6mpaWFrW0tASfe71eZWRkcPfrYxQIBHThhRe2eZuCpKQkrV69moviIazYJoHYEBV3v169erVGjx6tgoICDRo0SGeccYaeeuqp4PyHH36o+vp65ebmBseSk5M1bty4du/xM3fuXCUnJwcfGRkZ3f736Mm41xIiDdskgMMZLTL//Oc/tWjRIp166qlav369brjhBt1yyy16+umnJUn19fWSJJfLFfJ1LpcrOPdNpaWlam5uDj52797dvX+JHo57LSHSsE0COJzRIhMIBDRq1Cg98MADOuOMM3Tdddfp2muv1RNPPHHM75mQkKCkpKSQB45dXFycysrK2pybPXs2u/ARdmyTAA5n9P/4E088UaeffnrI2PDhw7Vr1y5JUmpqqiSpoaEh5DUNDQ3BOZhz6IwRINxGjx6tESNGhIyNHDlSo0aNMpQIgClGi0xOTo527twZMvbuu+8qMzNTkjR06FClpqZqw4YNwXmv16s33niD3cdhEggE2v3tt6ysjDIDY+bMmRPy/P777zeUBIBJRotMUVGRNm/erAceeEDvv/++li9frv/6r/9SYWGhJMmyLM2cOVP333+/Vq9ere3bt+uKK65QWlqaLr74YpPRYwYLKxGpDj87sa3nAGKD0SIzZswYvfjii1qxYoW+//3v67777tP8+fPldruDr7ntttt0880367rrrtOYMWP0+eefa926dXI6nQaTxw4WViJS3XjjjSHPD/0CBCC2GL2OTDgc6XnoaJ/H49GTTz7ZavyGG27Q1KlTDSRCrFu7dq3mzp3bary0tFSTJk0ykAhAV4uK68gg8gUCAa1YsaLNOY/HwxoZhJ3f79e8efPanJs3bx43MoVRVVVVKigoUFVVlekoMYMigw6xRgaRhhuZIlL5fD6Vl5eroaFB5eXl3EonTCgy6BBrZBBpuJEpIlVlZaUaGxslSY2NjfJ4PIYTxYZ40wHw7WzbNtrsS0tLVVpa2mr8jjvuMHamiNPplGVZRj4bZjkcDt12221trpG5/fbbuZEpjKirq5PH49Gh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OzICABDxKisr1djYKElqbGyUx+MxnCg2HNN1ZE4++WQ99dRT+tvf/qZ33nlHlZWVGjFiRKeCPPjgg7IsSzNnzgyO+Xw+FRYWKiUlRf369VN+fr4aGho69TkAAHS1uro6eTye4J2ubduWx+NRXV2d4WQ931EXGa/X2+Zj3759OnDgwDGFqKmp0ZNPPqmRI0eGjBcVFWnNmjVauXKlNm3apD179mjKlCnH9BkAAHQH27ZVUVGhQCAQMu73+1VRUREsN+geR11k+vfvr+OPP77Vo3///kpMTFRmZqZmzZrV6j9oez7//HO53W499dRTOv7444Pjzc3NWrJkiR555BFNnDhRWVlZWrp0qV5//XVt3rz5aGMDANAtamtrVVNT06qw2Latmpoa1dbWGkoWG466yCxbtkxpaWm64447tGrVKq1atUp33HGHBg8erEWLFum6667To48+esRX+S0sLNT555+v3NzckPGtW7fq4MGDIePDhg3TkCFDVF1d3e77tbS0tNpbBABAdxkyZIiSkpLanEtKStKQIUPCnCi2HPVZS08//bTKy8t1ySWXBMcmT56sESNG6Mknn9SGDRs0ZMgQzZkzR3fccUeH7/Xss8/qzTffVE1NTau5+vp69e7dW/379w8Zd7lcqq+vb/c9586dq9mzZx/dXwoAgGO0a9eudn9p9nq92rVrl0466aTwhoohR71H5vXXX9cZZ5zRavyMM84I7ik588wzg/dgas/u3bv1y1/+Uh6Pp0tPTystLVVzc3PwsXv37i57bwAAvikzM1Njxoxpc27s2LHKzMwMc6LYctRFJiMjQ0uWLGk1vmTJEmVkZEj6+rSzw9e7tGXr1q36+OOPNWrUKMXHxys+Pl6bNm3So48+qvj4eLlcLh04cEBNTU0hX9fQ0KDU1NR23zchIUFJSUkhDwAAuotlWZo6dWqbc1OnTuWq993sqA8tPfzwwyooKNDatWuDDXTLli3asWOHXnjhBUlfn4V06aWXdvg+P/3pT7V9+/aQsauuukrDhg3Tr3/9a2VkZKhXr17asGGD8vPzJUk7d+7Url27uIIwACBi2LatFStWyLKskAW/lmVp+fLlGjVqFGWmGx11kbnwwgu1c+dOPfHEE3r33XclSZMmTdKqVav0+eefS5JuuOGGb32f4447Tt///vdDxvr27auUlJTg+IwZM1RcXKwB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9PLLL2v8+PGm4/RYFBkAALrJ0qVL1dzcrFtuuUVxcSxL7Q4UGQAAELWohwAAIGpRZAAAQNSiyAAAgKhFkQEAAFGLIgMAAKIWRQZAxCgrK9MPf/jDbnnvjRs3yrKsLr0X00cffSTLsvT222932XsCODoUGQDH5Morr5RlWa0e5557ruloAGJIvOkAAKLXueeeq6VLl4aMJSQkGErTvoMHD5qOAKCbsEcGwDFLSEhQampqyOP444+XJFmWpSeffFIXXHCB+vTpo+HDh6u6ulrvv/++JkyYoL59++pHP/qRPvjgg1bv++STTyojI0N9+vTRJZdcoubm5uBcTU2NzjnnHA0cOFDJyck666yz9Oabb4Z8vWVZWrRokS688EL17dtXc+bMafUZX3zxhSZNmqScnJzg4abFixdr+PDhcjqdGjZsmBYuXBjyNX/72990xhlnyOl0avTo0Xrrrbc6+y0E0EkUGQDd5r777tMVV1yht99+W8OGDdMvfvELXX/99SotLdWWLVtk27ZuuummkK95//339fzzz2vNmjVat26d3nrrLd14443B+X379mn69Ol67bXXtHnzZp166qk677zztG/fvpD3KSsr089//nNt375dV199dchcU1OTzjnnHAUCAb388svq37+/PB6P7rnnHs2ZM0c7duzQAw88oLvvvltPP/20JOnzzz/XBRdcoNNPP11bt25VWVmZbr311m76zgE4YjYAHIPp06fbDofD7tu3b8hjzpw5tm3btiT7rrvuCr6+urralmQvWbIkOLZixQrb6XQGn8+aNct2OBx2XV1dcGzt2rV2XFycvXfv3jZz+P1++7jjjrPXrFkTHJNkz5w5M+R1f/nLX2xJ9o4dO+yRI0fa+fn5dktLS3D+5JNPtpcvXx7yNffdd5+dnZ1t27ZtP/nkk3ZKSor95ZdfBucXLVpkS7Lfeuutb/1+AegerJEBcMzOPvtsLVq0KGRswIABwT+PHDky+GeXyyVJGjFiRMiYz+eT1+tVUlKSJGnIkCEaPHhw8DXZ2dkKBALauXOnUlNT1dDQoLvuuksbN27Uxx9/LL/fry+++EK7du0KyTF69Og2M59zzjkaO3asnnvuOTkcDknS/v379cEHH2jGjBm69tprg6/96quvlJycLEnasWOHRo4cKafTGZINgFkUGQDHrG/fvjrllFPane/Vq1fwz5ZltTsWCASO+DOnT5+uxsZG/fa3v1VmZqYSEhKUnZ2tAwcOtMrWlvPPP18vvPCC3nnnnWCp+vzzzyVJTz31lMaNGxfy+kNlB0BkosgAiCi7du3Snj17lJaWJknavHmz4uLidNppp0mSqqqqtHDhQp133nmSpN27d+vTTz894vd/8MEH1a9fP/30pz/Vxo0bdfrpp8vlciktLU3//Oc/5Xa72/y64cOH65lnnpHP5wvuldm8eXNn/qoAugBFBsAxa2lpUX19fchYfHy8Bg4ceMzv6XQ6NX36dD388MPyer265ZZbdMkllyg1NVWSdOqpp+qZZ57R6NGj5fV69atf/UqJiYlH9RkPP/yw/H6/Jk6cqI0bN2rYsGGaPXu2brnlFiUnJ+vcc89VS0uLtmzZos8++0zFxcX6xS9+oTvvvFPXXnutSktL9dFHH+nhhx8+5r8ngK7BWUsAjtm6det04oknhjzOPPPMTr3nKaecoilTpui8887Tz372M40cOTLkNOglS5bos88+06hRo3T55Zfrlltu0aBBg476cyoqKnTJJZdo4sSJevfdd3XNNddo8eLFWrp0qUaMGKGzzjpLy5Yt09ChQyVJ/fr105o1a7R9+3adccYZuvPOO/XQQw916u8KoPMs27Zt0yEAAACOBXtkAABA1KLIAACAqEWRAQAAUYsiAwAAohZFBgAARC2KDAAAiFoUGQAAELUoMgAAIGpRZAAAQNSiyAAAgKhFkQEAAFHr/wPso3TV1lg/ZQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=\"Embarked\",y=\"Age\",data=data)\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b9dc8c4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2abb286e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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JeSMA2Xnzp35q7/6q9x0002D9i9dujTPPvtsnnnmmWzbti379+/P3XffXT9+4sSJtLe35+jRo9m+fXueeuqpbNy4MStXrjzzswAALilnFCjvvvtuFixYkK985Sv5hV/4hfr+3t7efPWrX82XvvSl3H777Zk5c2aefPLJbN++PS+99FKS5Nvf/nZee+21/M3f/E1uueWWzJs3L1/4wheybt26HD169NycFQBwUTujQOno6Eh7e3vmzJkzaH9XV1eOHTs2aP+MGTMyffr0dHZ2Jkk6Oztz4403prm5ub6mra0tfX192b1792mfr7+/P319fYM2AODSNXKoD3j66afzve99Lzt37nzPse7u7owePToTJkwYtL+5uTnd3d31Nf83Tk4dP3XsdFatWpXPf/7zQx0VALhIDekKyr59+/LHf/zH+frXv54rrrjifM30HsuXL09vb29927dv3wV7bgDgwhtSoHR1deXAgQP5wAc+kJEjR2bkyJHZtm1b1q5dm5EjR6a5uTlHjx7NwYMHBz2up6cnLS0tSZKWlpb3vKvn1Nen1vy4pqamVCqVQRsAcOkaUqDccccd2bVrV1599dX6duutt2bBggX1fx41alS2bt1af8yePXuyd+/e1Gq1JEmtVsuuXbty4MCB+potW7akUqmktbX1HJ0WAHAxG9I9KOPHj8/73//+QfvGjh2bSZMm1fcvWrQoy5Yty8SJE1OpVHL//fenVqtl9uzZSZK5c+emtbU19957b1avXp3u7u6sWLEiHR0daWpqOkenBQBczIZ8k+zPsmbNmgwfPjzz589Pf39/2trasn79+vrxESNGZPPmzVm8eHFqtVrGjh2bhQsX5tFHHz3XowAAF6lhAwMDA40eYqj6+vpSrVbT29t72d2Pct3DzzV6BC6gNx9rb/QIAOfMUH5++1s8AEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFAcgQIAFEegAADFESgAQHEECgBQHIECABRHoAAAxREoAEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFAcgQIAFEegAADFESgAQHEECgBQHIECABRHoAAAxREoAEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFCckY0eAID/cd3DzzV6BC6gNx9rb/QIRXMFBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAijOkQHniiSdy0003pVKppFKppFar5fnnn68fP3LkSDo6OjJp0qSMGzcu8+fPT09Pz6DvsXfv3rS3t2fMmDGZMmVKHnzwwRw/fvzcnA0AcEkYUqBcc801eeyxx9LV1ZVXXnklt99+e+68887s3r07SbJ06dI8++yzeeaZZ7Jt27bs378/d999d/3xJ06cSHt7e44ePZrt27fnqaeeysaNG7Ny5cpze1YAwEVt2MDAwMDZfIOJEyfmL/7iL3LPPfdk8uTJ2bRpU+65554kyeuvv54bbrghnZ2dmT17dp5//vl8+MMfzv79+9Pc3Jwk2bBhQx566KG89dZbGT169M/1nH19falWq+nt7U2lUjmb8S861z38XKNH4AJ687H2Ro/ABeT1fXm5HF/fQ/n5fcb3oJw4cSJPP/10Dh8+nFqtlq6urhw7dixz5sypr5kxY0amT5+ezs7OJElnZ2duvPHGepwkSVtbW/r6+upXYU6nv78/fX19gzYA4NI15EDZtWtXxo0bl6ampnzmM5/JN77xjbS2tqa7uzujR4/OhAkTBq1vbm5Od3d3kqS7u3tQnJw6furYT7Jq1apUq9X6Nm3atKGODQBcRIYcKL/yK7+SV199NS+//HIWL16chQsX5rXXXjsfs9UtX748vb299W3fvn3n9fkAgMYaOdQHjB49Or/0S7+UJJk5c2Z27tyZL3/5y/noRz+ao0eP5uDBg4OuovT09KSlpSVJ0tLSkh07dgz6fqfe5XNqzek0NTWlqalpqKMCABeps/4clJMnT6a/vz8zZ87MqFGjsnXr1vqxPXv2ZO/evanVakmSWq2WXbt25cCBA/U1W7ZsSaVSSWtr69mOAgBcIoZ0BWX58uWZN29epk+fnkOHDmXTpk35p3/6p3zrW99KtVrNokWLsmzZskycODGVSiX3339/arVaZs+enSSZO3duWltbc++992b16tXp7u7OihUr0tHR4QoJAFA3pEA5cOBAPvGJT+RHP/pRqtVqbrrppnzrW9/K7//+7ydJ1qxZk+HDh2f+/Pnp7+9PW1tb1q9fX3/8iBEjsnnz5ixevDi1Wi1jx47NwoUL8+ijj57bswIALmpn/TkojeBzULhcXI6fk3A58/q+vFyOr+8L8jkoAADni0ABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKM6QAmXVqlX59V//9YwfPz5TpkzJXXfdlT179gxac+TIkXR0dGTSpEkZN25c5s+fn56enkFr9u7dm/b29owZMyZTpkzJgw8+mOPHj5/92QAAl4QhBcq2bdvS0dGRl156KVu2bMmxY8cyd+7cHD58uL5m6dKlefbZZ/PMM89k27Zt2b9/f+6+++768RMnTqS9vT1Hjx7N9u3b89RTT2Xjxo1ZuXLluTsrAOCiNmxgYGDgTB/81ltvZcqUKdm2bVtuu+229Pb2ZvLkydm0aVPuueeeJMnrr7+eG264IZ2dnZk9e3aef/75fPjDH87+/fvT3NycJNmwYUMeeuihvPXWWxk9evTPfN6+vr5Uq9X09vamUqmc6fgXpesefq7RI3ABvflYe6NH4ALy+r68XI6v76H8/D6re1B6e3uTJBMnTkySdHV15dixY5kzZ059zYwZMzJ9+vR0dnYmSTo7O3PjjTfW4yRJ2tra0tfXl927d5/2efr7+9PX1zdoAwAuXWccKCdPnswDDzyQ3/qt38r73//+JEl3d3dGjx6dCRMmDFrb3Nyc7u7u+pr/Gyenjp86djqrVq1KtVqtb9OmTTvTsQGAi8AZB0pHR0f+7d/+LU8//fS5nOe0li9fnt7e3vq2b9++8/6cAEDjjDyTBy1ZsiSbN2/Oiy++mGuuuaa+v6WlJUePHs3BgwcHXUXp6elJS0tLfc2OHTsGfb9T7/I5tebHNTU1pamp6UxGBQAuQkO6gjIwMJAlS5bkG9/4Rr773e/m+uuvH3R85syZGTVqVLZu3Vrft2fPnuzduze1Wi1JUqvVsmvXrhw4cKC+ZsuWLalUKmltbT2bcwEALhFDuoLS0dGRTZs25e///u8zfvz4+j0j1Wo1V155ZarVahYtWpRly5Zl4sSJqVQquf/++1Or1TJ79uwkydy5c9Pa2pp77703q1evTnd3d1asWJGOjg5XSQCAJEMMlCeeeCJJ8ru/+7uD9j/55JP55Cc/mSRZs2ZNhg8fnvnz56e/vz9tbW1Zv359fe2IESOyefPmLF68OLVaLWPHjs3ChQvz6KOPnt2ZAACXjCEFys/zkSlXXHFF1q1bl3Xr1v3ENddee23+8R//cShPDQBcRvwtHgCgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOIIFACgOAIFACiOQAEAiiNQAIDiCBQAoDhDDpQXX3wxH/nIRzJ16tQMGzYs3/zmNwcdHxgYyMqVK3P11VfnyiuvzJw5c/KDH/xg0Jp33nknCxYsSKVSyYQJE7Jo0aK8++67Z3UiAMClY8iBcvjw4dx8881Zt27daY+vXr06a9euzYYNG/Lyyy9n7NixaWtry5EjR+prFixYkN27d2fLli3ZvHlzXnzxxdx3331nfhYAwCVl5FAfMG/evMybN++0xwYGBvL4449nxYoVufPOO5Mkf/3Xf53m5uZ885vfzMc+9rH8+7//e1544YXs3Lkzt956a5LkL//yL/MHf/AH+eIXv5ipU6eexekAAJeCc3oPyhtvvJHu7u7MmTOnvq9arWbWrFnp7OxMknR2dmbChAn1OEmSOXPmZPjw4Xn55ZdP+337+/vT19c3aAMALl3nNFC6u7uTJM3NzYP2Nzc31491d3dnypQpg46PHDkyEydOrK/5catWrUq1Wq1v06ZNO5djAwCFuSjexbN8+fL09vbWt3379jV6JADgPDqngdLS0pIk6enpGbS/p6enfqylpSUHDhwYdPz48eN555136mt+XFNTUyqVyqANALh0ndNAuf7669PS0pKtW7fW9/X19eXll19OrVZLktRqtRw8eDBdXV31Nd/97ndz8uTJzJo161yOAwBcpIb8Lp533303P/zhD+tfv/HGG3n11VczceLETJ8+PQ888ED+7M/+LL/8y7+c66+/Pn/6p3+aqVOn5q677kqS3HDDDfnQhz6UT3/609mwYUOOHTuWJUuW5GMf+5h38AAASc4gUF555ZX83u/9Xv3rZcuWJUkWLlyYjRs35k/+5E9y+PDh3HfffTl48GA++MEP5oUXXsgVV1xRf8zXv/71LFmyJHfccUeGDx+e+fPnZ+3atefgdACAS8GwgYGBgUYPMVR9fX2pVqvp7e297O5Hue7h5xo9AhfQm4+1N3oELiCv78vL5fj6HsrP74viXTwAwOVFoAAAxREoAEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFAcgQIAFEegAADFESgAQHEECgBQHIECABRHoAAAxREoAEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFAcgQIAFEegAADFESgAQHEECgBQHIECABRHoAAAxREoAEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFAcgQIAFEegAADFESgAQHEECgBQHIECABRHoAAAxREoAEBxBAoAUByBAgAUR6AAAMURKABAcQQKAFAcgQIAFEegAADFESgAQHEECgBQHIECABRHoAAAxREoAEBxBAoAUByBAgAUp6GBsm7dulx33XW54oorMmvWrOzYsaOR4wAAhWhYoPzt3/5tli1blkceeSTf+973cvPNN6etrS0HDhxo1EgAQCEaFihf+tKX8ulPfzqf+tSn0tramg0bNmTMmDH52te+1qiRAIBCjGzEkx49ejRdXV1Zvnx5fd/w4cMzZ86cdHZ2vmd9f39/+vv761/39vYmSfr6+s7/sIU52f//Gj0CF9Dl+O/45czr+/JyOb6+T53zwMDAz1zbkEB5++23c+LEiTQ3Nw/a39zcnNdff/0961etWpXPf/7z79k/bdq08zYjlKD6eKMnAM6Xy/n1fejQoVSr1Z+6piGBMlTLly/PsmXL6l+fPHky77zzTiZNmpRhw4Y1cDIuhL6+vkybNi379u1LpVJp9DjAOeT1fXkZGBjIoUOHMnXq1J+5tiGBctVVV2XEiBHp6ekZtL+npyctLS3vWd/U1JSmpqZB+yZMmHA+R6RAlUrFf8DgEuX1ffn4WVdOTmnITbKjR4/OzJkzs3Xr1vq+kydPZuvWranVao0YCQAoSMN+xbNs2bIsXLgwt956a37jN34jjz/+eA4fPpxPfepTjRoJAChEwwLlox/9aN56662sXLky3d3dueWWW/LCCy+858ZZaGpqyiOPPPKeX/MBFz+vb36SYQM/z3t9AAAuIH+LBwAojkABAIojUACA4ggUAKA4AgUAKM5F8VH3XF7efvvtfO1rX0tnZ2e6u7uTJC0tLfnN3/zNfPKTn8zkyZMbPCEA55srKBRl586ded/73pe1a9emWq3mtttuy2233ZZqtZq1a9dmxowZeeWVVxo9JnCe7Nu3L3/0R3/U6DEogM9BoSizZ8/OzTffnA0bNrznD0EODAzkM5/5TP71X/81nZ2dDZoQOJ++//3v5wMf+EBOnDjR6FFoML/ioSjf//73s3HjxtP+lephw4Zl6dKl+bVf+7UGTAacC//wD//wU4//53/+5wWahNIJFIrS0tKSHTt2ZMaMGac9vmPHDn8OAS5id911V4YNG5afdvH+dP+DwuVHoFCUz372s7nvvvvS1dWVO+64ox4jPT092bp1a77yla/ki1/8YoOnBM7U1VdfnfXr1+fOO+887fFXX301M2fOvMBTUSKBQlE6Ojpy1VVXZc2aNVm/fn3999AjRozIzJkzs3HjxvzhH/5hg6cEztTMmTPT1dX1EwPlZ11d4fLhJlmKdezYsbz99ttJkquuuiqjRo1q8ETA2frnf/7nHD58OB/60IdOe/zw4cN55ZVX8ju/8zsXeDJKI1AAgOL4HBQAoDgCBQAojkABAIojUACA4ggUAKA4AgUAKI5AAQCKI1AAgOL8f09Kl8A8tZ0iAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#we will get the unique value count\n",
    "data['Survived'].value_counts().plot(kind='bar')\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1558096c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Embarked'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['Embarked'].value_counts().plot(kind='pie')\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0047d4f8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b5259fa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9df0c244",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18a61396",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7862ec0a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          891 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Embarked     891 non-null    object \n",
      "dtypes: float64(2), int64(5), object(4)\n",
      "memory usage: 76.7+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "53af006c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "male      577\n",
       "female    314\n",
       "Name: Sex, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Sex'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ecaf6b11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     male\n",
       "0       1\n",
       "1       0\n",
       "2       0\n",
       "3       0\n",
       "4       1\n",
       "..    ...\n",
       "886     1\n",
       "887     0\n",
       "888     0\n",
       "889     1\n",
       "890     1\n",
       "\n",
       "[891 rows x 1 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Feature Encoding -->converting categorical data to numerical values\n",
    "sex=pd.get_dummies(data['Sex'],drop_first=True)\n",
    "sex\n",
    "#do for pclass,Embarked"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6dc957ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     2  3\n",
       "0    0  1\n",
       "1    0  0\n",
       "2    0  1\n",
       "3    0  0\n",
       "4    0  1\n",
       "..  .. ..\n",
       "886  1  0\n",
       "887  0  0\n",
       "888  0  1\n",
       "889  0  0\n",
       "890  0  1\n",
       "\n",
       "[891 rows x 2 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pclass=pd.get_dummies(data['Pclass'],drop_first=True)\n",
    "pclass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "df65227b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Q  S\n",
       "0    0  1\n",
       "1    0  0\n",
       "2    0  1\n",
       "3    0  1\n",
       "4    0  1\n",
       "..  .. ..\n",
       "886  0  1\n",
       "887  0  1\n",
       "888  0  1\n",
       "889  0  0\n",
       "890  1  0\n",
       "\n",
       "[891 rows x 2 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "embarked=pd.get_dummies(data['Embarked'],drop_first=True)\n",
    "embarked"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "05029fb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>male</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>Q</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex        Age  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.000000   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.000000   \n",
       "2                               Heikkinen, Miss. Laina  female  26.000000   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.000000   \n",
       "4                             Allen, Mr. William Henry    male  35.000000   \n",
       "..                                                 ...     ...        ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.000000   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.000000   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female  29.699118   \n",
       "889                              Behr, Mr. Karl Howell    male  26.000000   \n",
       "890                                Dooley, Mr. Patrick    male  32.000000   \n",
       "\n",
       "     SibSp  Parch            Ticket     Fare Embarked  male  2  3  Q  S  \n",
       "0        1      0         A/5 21171   7.2500        S     1  0  1  0  1  \n",
       "1        1      0          PC 17599  71.2833        C     0  0  0  0  0  \n",
       "2        0      0  STON/O2. 3101282   7.9250        S     0  0  1  0  1  \n",
       "3        1      0            113803  53.1000        S     0  0  0  0  1  \n",
       "4        0      0            373450   8.0500        S     1  0  1  0  1  \n",
       "..     ...    ...               ...      ...      ...   ... .. .. .. ..  \n",
       "886      0      0            211536  13.0000        S     1  1  0  0  1  \n",
       "887      0      0            112053  30.0000        S     0  0  0  0  1  \n",
       "888      1      2        W./C. 6607  23.4500        S     0  0  1  0  1  \n",
       "889      0      0            111369  30.0000        C     1  0  0  0  0  \n",
       "890      0      0            370376   7.7500        Q     1  0  1  1  0  \n",
       "\n",
       "[891 rows x 16 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_data=pd.concat([data,sex,pclass,embarked],axis='columns')\n",
    "final_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "d40f6861",
   "metadata": {},
   "outputs": [],
   "source": [
    "final_data.drop(columns=['PassengerId','Sex','Name',\n",
    "                               'Pclass',\n",
    "                               'Ticket',\n",
    "                               'Embarked'],inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2806c56b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>male</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived        Age  SibSp  Parch     Fare  male  2  3  Q  S\n",
       "0           0  22.000000      1      0   7.2500     1  0  1  0  1\n",
       "1           1  38.000000      1      0  71.2833     0  0  0  0  0\n",
       "2           1  26.000000      0      0   7.9250     0  0  1  0  1\n",
       "3           1  35.000000      1      0  53.1000     0  0  0  0  1\n",
       "4           0  35.000000      0      0   8.0500     1  0  1  0  1\n",
       "..        ...        ...    ...    ...      ...   ... .. .. .. ..\n",
       "886         0  27.000000      0      0  13.0000     1  1  0  0  1\n",
       "887         1  19.000000      0      0  30.0000     0  0  0  0  1\n",
       "888         0  29.699118      1      2  23.4500     0  0  1  0  1\n",
       "889         1  26.000000      0      0  30.0000     1  0  0  0  0\n",
       "890         0  32.000000      0      0   7.7500     1  0  1  1  0\n",
       "\n",
       "[891 rows x 10 columns]"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "907d7e4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>male</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived   Age  SibSp  Parch     Fare  male  2  3  Q  S\n",
       "0         0  22.0      1      0   7.2500     1  0  1  0  1\n",
       "1         1  38.0      1      0  71.2833     0  0  0  0  0\n",
       "2         1  26.0      0      0   7.9250     0  0  1  0  1\n",
       "3         1  35.0      1      0  53.1000     0  0  0  0  1\n",
       "4         0  35.0      0      0   8.0500     1  0  1  0  1"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "5d8b125f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>male</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.75</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived        Age  SibSp  Parch   Fare  male  2  3  Q  S\n",
       "886         0  27.000000      0      0  13.00     1  1  0  0  1\n",
       "887         1  19.000000      0      0  30.00     0  0  0  0  1\n",
       "888         0  29.699118      1      2  23.45     0  0  1  0  1\n",
       "889         1  26.000000      0      0  30.00     1  0  0  0  0\n",
       "890         0  32.000000      0      0   7.75     1  0  1  1  0"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_data.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8ebb402",
   "metadata": {},
   "source": [
    "# building the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "12c68b14",
   "metadata": {},
   "outputs": [],
   "source": [
    "#training and testing part\n",
    "X=final_data.drop('Survived',axis=1)\n",
    "y=final_data[\"Survived\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5afaed03",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Age', 'SibSp', 'Parch', 'Fare', 'male', 2, 3, 'Q', 'S'], dtype='object')"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "256d6850",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1e67554e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "53d0348e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "55d80df2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logmodel = LogisticRegression()\n",
    "logmodel.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "aeb83fe5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictors\n",
    "predictions=logmodel.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "349c6e83",
   "metadata": {},
   "outputs": [],
   "source": [
    "#final metrics calculation\n",
    "from sklearn.metrics import plot_confusion_matrix,accuracy_score,confusion_matrix,classification_report,roc_curve\n",
    "#print(accuracy_score(y_test,predictions)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "7475c153",
   "metadata": {},
   "outputs": [],
   "source": [
    "#print(confusion_matrix(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "b23ecf9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#print(classification_report(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "1c894976",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1726a11e1c8>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Ha+TfdeuXbVly5Y6+//617/qqquu8klQAAD4nI0re69X0Js2bZoyMjJ06NAhuVwuvfXWW9q7d69WrFihtWvXNkSMAACYZ+O33nld2Q8ePFhr1qzRP//5TzVr1kzTpk3Tnj17tGbNGv385z9viBgBAIAJ57U2/o033qj169f7OhYAABqMnV9xe94vwtm+fbv27Nkj6dQ4fo8ePXwWFAAAPsdb7+rv4MGDuuuuu/Svf/1LsbGxkqTS0lJdd911ev3113XJJZf4OkYAAGCC12P2999/v2pqarRnzx6VlJSopKREe/bskcvl0v33398QMQIAYN7pCXpmtgDldWW/adMmbd26VZ06dXLv69SpkxYtWqQbb7zRp8EBAOArDuPUZub6QOV1ZZ+UlHTGxXOcTqcSExN9EhQAAD7XyM/ZO51OZWZmKjk5WREREbr00ks1e/ZsGf8x088wDE2bNk2tWrVSRESE0tLStG/fPpM/aF1eJ/unnnpKDz/8sLZv3+7et337do0dO1ZPP/20T4MDACBQPfnkk1qyZImee+457dmzR08++aTmz5+vRYsWuc+ZP3++srOztXTpUuXm5qpZs2bq16+fz98qW682fvPmzeVw/DhWUVFRoV69eikk5NTltbW1CgkJ0X333achQ4b4NEAAAHzCR4vqlJWVeewOCwtTWFhYndO3bt2qwYMH6xe/+IUkqV27dnrttdf00UcfnbqdYWjhwoV64oknNHjwYEmn3j8THx+vnJwcDR8+/Pxj/S/1SvYLFy702RcCAOAXPnr0LikpyWP39OnTNWPGjDqnX3fddXrhhRf01Vdf6bLLLtPu3bv1wQcf6Nlnn5Uk5efnq6ioSGlpae5rYmJi1KtXL23btq3xk31GRobPvhAAgEBWUFCg6Oho9+czVfWSNGXKFJWVlalz584KDg6W0+nU3LlzlZ6eLkkqKiqSJMXHx3tcFx8f7z7mK+e9qI4kVVZWqrq62mPff/4HAADgguGjyj46OrpeuW716tVauXKlVq1apcsvv1y7du3SuHHjlJiY2OhFtNfJvqKiQpMnT9bq1av1/fff1znudDp9EhgAAD7VyCvoPfbYY5oyZYq7Hd+tWzd9++23ysrKUkZGhhISEiRJxcXFatWqlfu64uJide/e3USgdXk9G3/SpEnauHGjlixZorCwML300kuaOXOmEhMTtWLF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51ZXVys0NNQn3xsXF+eT+wCAr1DZw7LCwsKUkJCgtm3b6oEHHlBaWpreeecdST+23ufOnavExER16tRJklRQUKBhw4YpNjZWcXFxGjx4sL755hv3PZ1OpyZMmKDY2Fi1aNFCkyZN0n+/XuK/2/hVVVWaPHmykpKSFBYWpg4dOujll1/WN9984375SvPmzeVwOHTvvfdKOvUK4aysLCUnJysiIkIpKSn661//6vE97777ri677DJFRESob9++HnHW1+TJk3XZZZepadOmat++vTIzM1VTU1PnvOeff15JSUlq2rSphg0bpuPHj3scf+mll9SlSxeFh4erc+fO+uMf/+h1LAAaDskethEREaHq6mr35w0bNmjv3r1av3691q5dq5qaGvXr109RUVHasmWL/vWvfykyMlL9+/d3X/fMM89o+fLleuWVV/TBBx+opKREb7/99jm/95577tFrr72m7Oxs7dmzR88//7wiIyOVlJSkN998U5K0d+9eHT58WH/4wx8kSVlZWVqxYoWWLl2qzz//XOPHj9fdd9+tTZs2STr1R8nQoUM1aNAg7dq1S/fff7+mTJni9X+TqKgoLV++XF988YX+8Ic/6MUXX9SCBQs8zsnLy9Pq1au1Zs0arVu3Tp988okefPBB9/GVK1dq2rRpmjt3rvbs2aN58+YpMzNTr776qtfxAGggBmBBGRkZxuDBgw3DMAyXy2WsX7/eCAsLMyZOnOg+Hh8fb1RVVbmv+dOf/mR06tTJcLlc7n1VVVVGRESE8Y9//MMwDMNo1aqVMX/+fPfxmpoa45JLLnF/l2EYRu/evY2xY8cahmEYe/fuNSQZ69evP2Oc7733niHJOHbsmHtfZWWl0bRpU2Pr1q0e544cOdK46667DMMwjKlTpxpdu3b1OD558uQ69/pvkoy33377rMefeuopo0ePHu7P06dPN4KDg42DBw+69/397383goKCjMOHDxuGYRiXXnqpsWrVKo/7zJ4920hNTTUMwzDy8/MNScYnn3xy1u8F0LAYs4dlrV27VpGRkaqpqZHL5dL//M//aMaMGe7j3bp18xin3717t/Ly8hQVFeVxn8rKSu3fv1/Hjx/X4cOH1atXL/exkJAQ9ezZs04r/7Rdu3YpODhYvXv3rnfceXl5OnnypH7+85977K+urtZVV10lSdqzZ49HHJKUmppa7+847Y033lB2drb279+v8vJy1dbWKjo62uOcNm3aqHXr1h7f43K5tHfvXkVFRWn//v0aOXKkRo0a5T6ntrZWMTExXscDoGGQ7GFZffv21ZIlSxQaGqrExESFhHj+ujdr1szjc3l5uXr06KGVK1fWudfFF198XjFERER4fU15ebkk6W9/+5tHkpVOzUPwlW3btik9PV0zZ85Uv379FBMTo9dff13PPPOM17G++OKLdf74CA4O9lmsAMwh2cOymjVrpg4dOtT7/KuvvlpvvPGGWrZsWae6Pa1Vq1bKzc3VTTfdJOlUBbtjxw5dffXVZzy/W7ducrlc2rRpk9LS0uocP91ZcDqd7n1du3ZVWFiYDhw4cNaOQJcuXdyTDU/78MMPf/qH/A9bt25V27Zt9fjjj7v3ffvtt3XOO3DggAoLC5WYmOj+nqCgIHXq1Enx8fFKTEzU119/rfT0dK++H0DjYYIe8G/p6em66KKLNHjwYG3ZskX5+fl6//339cgjj+jgwYOSpLFjx+r3v/+9cnJy9OWXX+rBBx885zPy7dq1U0ZGhu677z7l5OS477l69WpJUtu2beVwOLR27Vp99913Ki8vV1RUlCZOnKjx48fr1Vdf1f79+7Vz504tWrTIPentd7/7nfbt26fHHntMe/fu1apVq7R8+XKvft6OHTvqwIEDev3117V//35lZ2efcbJheHi4MjIytHv3bm3ZskWPPPKIhg0bpoSEBEnSzJkzlZWVpezsbH311Vf69NNPtWzZMj377LNexQOg4ZDsgX9r2rSpNm/erDZt2mjo0KHq0qWLRo4cqcrKSnel/+ijj+o3v/mNMjIylJqaqqioKP3yl788532XLFmiO++8Uw8++KA6d+6sUaNGqaKiQpLUunVrzZw5U1OmTFF8fLzGjBkjSZo9e7YyMzOVlZWlLl26qH///vrb3/6m5ORkSafG0d98803l5OQoJSVFS5cu1bx587z6eW+//XaNHz9eY8aMUffu3bV161ZlZmbWOa9Dhw4aOnSoBg4cqFtvvVVXXnmlx6N1999/v1566SUtW7ZM3bp1U+/evbV8+XJ3rAD8z2GcbWYRAACwBCp7AAAsjmQPAIDFkewBALA4kj0AABZHsgcAwOJI9gAAWBzJHgAAiyPZAwBgcSR7AAAsjmQPAIDFkewBALC4/weA9CoQiuzLTAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plot confusion_matrix(model,X_train,y_train,values_format='d')\n",
    "plot_confusion_matrix(logmodel,X_test,y_test,values_format='d')\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "6302d554",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0], dtype=int64)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logmodel.predict([[32,0,0,7.75,1,0,1,1,0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ee74d4e3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ca49689",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
