import requests
import pandas
import matplotlib.pyplot as plt

url_target = 'https://gql.tokopedia.com/graphql/ShopProducts'
header = {
    'authority': 'gql.tokopedia.com',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Cookie': 'ak_bmsc=CBF3D06C613C517CA750319DD24EE452~000000000000000000000000000000~YAAQLHsbuDvS30+PAQAARnvAVhdrtFBjFHosT+WgBZGEwV1yjok5n9M+rQpTHNqo9bCgScINJRuBRDk4+8TEEYlTFBVyeNESgkzqJMzN0h0cw29vEb0GARPLfJ8LqvmiRkd8OnZ65xuBFQDLxe59G+kuo+PuY8Ug7PRTutxXbHG8WyVE2N9p3QUDawKmy2k8drGYJD9TLQVdKsNqz3RkYtA+5adJRpl1TL/n5LpL1RCJN33Dq5KFvvnUZV8/fbw7kBKkPKLcn9JF/E7DPm/o9DXr5ZADB2WJVFWbbNxMxPvK0wF3nFlaGkddVgMqEjy8SWSvr46LsQ8wWKv+ch4TdsPfQpPaXMFGX2B4a3P484f1TpIj+ZXsxYAS7eT3m9kBwO+PnQ63CUBG+uktXIMhGn8Pn3hlTQbBssqraRavGGbkOrHRNzjNe4qmGBl2ycRSysckbl4=; _gcl_au=1.1.1425200151.1715147429; _UUID_NONLOGIN_=afd49ee45f10a88681eef198be054b33; _UUID_CAS_=53e7c273-0730-4d56-b2ab-3f5647cfaf4f; _CASE_=7b22644964223a323237342c22614964223a302c226c626c223a224a616b61727461205075736174222c22634964223a3137362c226c6f6e67223a22222c226c6174223a22222c2270436f223a22222c22774964223a302c22734964223a302c227354797065223a22222c22776873223a225b5d222c226c557064223a22227d; _gid=GA1.2.91102571.1715147430; _SID_Tokopedia_=P72a0hB745XxBh0UZMfpXN5uFydVolMCakKIOZFYqL_GNQf8SHWa2_3YNpTvDmL-MmChTsWEKq9uSTJKzhWzaYvo7GELu5AmFdbLRNlWE_7cNbj5m6re5vFiLW0hUIed; DID=d494f9ccc79d907a10258be16f3da4ddec3e9080b7927dd885c70d276b1421ab977eca5a5fd7c6e8771109fa0f4239f6; DID_JS=ZDQ5NGY5Y2NjNzlkOTA3YTEwMjU4YmUxNmYzZGE0ZGRlYzNlOTA4MGI3OTI3ZGQ4ODVjNzBkMjc2YjE0MjFhYjk3N2VjYTVhNWZkN2M2ZTg3NzExMDlmYTBmNDIzOWY247DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=; _fbp=fb.1.1715147434671.114827797; _dc_gtm_UA-9801603-1=1; _abck=A668ADB99A0A20FB9CE0B5938D510D94~0~YAAQZHU2F2CVnFaPAQAAjKL5Vgtl3n+evMqhNWuA3sOLz6SJO9AM0NckL0aIq6SUKfrs5jYTF417qymVsZCAfHw7yuuS2sRraR2gw4ORAPa1KRwMA8qH8yPL/CLMZlYn1IYDA8m9YU4W0XAriQemALmAJ9jxNrtr1ViKEuqccf6nCrlJiDbMpGdVLzX3U4x4ITCcPYGsVE/9z1GhDTmXHoPRGBsxRUMDTYXqB9iX7HzJlu8lzjS+2BgKS8Y0GKCuLTB5Mha9JDL/7o+MAJ+/JZJ1qEM4danSMpESf2DP55M5Ai6u7CHHzzf9u261WV7uB2UeOMB86Llo5ohyE2pKFUgn8Ibw30pb0tdh/AByZqXjzhLs/3vJHw153dR7uv9uw8gYQIiFCUesaXFuZ14Ed814P/TTMxrOf1kq~-1~-1~-1; bm_sv=128D5A69B64C294D5639EA06156EFE49~YAAQZHU2F2GVnFaPAQAAjKL5Vhfl8viNqYBc6EU6oXMUakZAgp+u9nRDw+BCLD+nnEe83oTLbXON3JVbEoJC2Sx/VHQIAXMJGV8q0HKqCXacpdG7EAhmoPRF+8muMcGmE4gIRgKI303K4qp/RJy43GxWSSJQzm7fM1wCXxBajcQdTDF6rJZnjzcLfcyklemM7xCnLFmOxArJXwP/r8lfSQhJH6JURgl/BKFm+C4+nO1q08Id8ZWDJ9ZTcHwr5t4ky8Jb~1; bm_sz=F61D4E14E66114E954B4BD175FE32FFB~YAAQZHU2F2KVnFaPAQAAjKL5VhfvAfCINDSX/9pBtxZw6RKr+ChTxByZupyqgmMsiysyxTI6MiYM/D3PNWyp8S7HuZrnzmrEiTSXaVKNVeky5lyo/WVsNKG7yRJ3ZUwNtLwX3295za6lEfQOuN4LIa8LBYemAIHqyymeOJL1bAy2gigA+O3jYwD60slzfQvelGiTdKsTo47X79q+hcTc/QMm1Rlz35CURc/PkVVQpKGCDDW9sEocVeDqOxchmNGxj/s0WpdYiBRRyEvzlASgg+sVhHB+fMWsepTpijqr912S3/26H98BusdrJbC8udosAZUsUNXbJoddtgE6KB2Pb+uTDdV+aCCJ4dFvZ1Bxpo7pEpoNZxjqtgfKd9UK9NnTG8MRmYoBWLWaVJBXOsQx1/HOBtKCeb9nubrQ1anja3LA33NzlnfwbmxkhnveznS5khfmViClsb1SNtAZhA==~4474417~3160131; ISID=%7B%22www.tokopedia.com%22%3A%22d3d3LnRva29wZWRpYS5jb20%3D.843b2ffcfb0544e9cbff538cc71df5ce.1715149325099.1715149325099.1715151152709.5%22%7D; AMP_TOKEN=%24NOT_FOUND; _dc_gtm_UA-126956641-6=1; _ga_70947XW48P=GS1.1.1715149325.2.1.1715151152.58.0.0; _ga=GA1.1.1809706914.1715147429',
    'Origin': 'https://www.tokopedia.com',
    'Referer': 'https://www.tokopedia.com/tokokaoskatun',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'X-Device': 'default_v3',
    'X-Source': 'tokopedia-lite',
    'X-Tkpd-Lite-Service': 'zeus',
    'X-Version': '21ff5f6',
    'accept': '*/*',
    'content-type': 'application/json',
    'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"'
}
query = f'[{{"operationName":"ShopProducts","variables":{{"source":"shop","sid":"2354714","page":1,"perPage":80,"etalaseId":"etalase","sort":1,"user_districtId":"2274","user_cityId":"176","user_lat":"","user_long":""}},"query":"query ShopProducts($sid: String\u0021, $source: String, $page: Int, $perPage: Int, $keyword: String, $etalaseId: String, $sort: Int, $user_districtId: String, $user_cityId: String, $user_lat: String, $user_long: String) {{\\n  GetShopProduct(shopID: $sid, source: $source, filter: {{page: $page, perPage: $perPage, fkeyword: $keyword, fmenu: $etalaseId, sort: $sort, user_districtId: $user_districtId, user_cityId: $user_cityId, user_lat: $user_lat, user_long: $user_long}}) {{\\n    status\\n    errors\\n    links {{\\n      prev\\n      next\\n      __typename\\n    }}\\n    data {{\\n      name\\n      product_url\\n      product_id\\n      price {{\\n        text_idr\\n        __typename\\n      }}\\n      primary_image {{\\n        original\\n        thumbnail\\n        resize300\\n        __typename\\n      }}\\n      flags {{\\n        isSold\\n        isPreorder\\n        isWholesale\\n        isWishlist\\n        __typename\\n      }}\\n      campaign {{\\n        discounted_percentage\\n        original_price_fmt\\n        start_date\\n        end_date\\n        __typename\\n      }}\\n      label {{\\n        color_hex\\n        content\\n        __typename\\n      }}\\n      label_groups {{\\n        position\\n        title\\n        type\\n        url\\n        __typename\\n      }}\\n      badge {{\\n        title\\n        image_url\\n        __typename\\n      }}\\n      stats {{\\n        reviewCount\\n        rating\\n        averageRating\\n        __typename\\n      }}\\n      category {{\\n        id\\n        __typename\\n      }}\\n      __typename\\n    }}\\n    __typename\\n  }}\\n}}\\n"}}]'
query = f'[{{"operationName":"ShopProducts","variables":{{"source":"shop","sid":"2668215","page":1,"perPage":80,"etalaseId":"etalase","sort":1,"user_districtId":"2274","user_cityId":"176","user_lat":"","user_long":""}},"query":"query ShopProducts($sid: String\u0021, $source: String, $page: Int, $perPage: Int, $keyword: String, $etalaseId: String, $sort: Int, $user_districtId: String, $user_cityId: String, $user_lat: String, $user_long: String) {{\\n  GetShopProduct(shopID: $sid, source: $source, filter: {{page: $page, perPage: $perPage, fkeyword: $keyword, fmenu: $etalaseId, sort: $sort, user_districtId: $user_districtId, user_cityId: $user_cityId, user_lat: $user_lat, user_long: $user_long}}) {{\\n    status\\n    errors\\n    links {{\\n      prev\\n      next\\n      __typename\\n    }}\\n    data {{\\n      name\\n      product_url\\n      product_id\\n      price {{\\n        text_idr\\n        __typename\\n      }}\\n      primary_image {{\\n        original\\n        thumbnail\\n        resize300\\n        __typename\\n      }}\\n      flags {{\\n        isSold\\n        isPreorder\\n        isWholesale\\n        isWishlist\\n        __typename\\n      }}\\n      campaign {{\\n        discounted_percentage\\n        original_price_fmt\\n        start_date\\n        end_date\\n        __typename\\n      }}\\n      label {{\\n        color_hex\\n        content\\n        __typename\\n      }}\\n      label_groups {{\\n        position\\n        title\\n        type\\n        url\\n        __typename\\n      }}\\n      badge {{\\n        title\\n        image_url\\n        __typename\\n      }}\\n      stats {{\\n        reviewCount\\n        rating\\n        averageRating\\n        __typename\\n      }}\\n      category {{\\n        id\\n        __typename\\n      }}\\n      __typename\\n    }}\\n    __typename\\n  }}\\n}}\\n"}}]'
response = requests.post(url_target, headers=header, data=query)
products = response.json()[0]['data']['GetShopProduct']['data']

daftar_product = []
for data in products:
   nama_product = data['name']
   foto_product = data['primary_image']['original']
   harga_product = data['price']['text_idr']
   diskon_product = data['campaign']['discounted_percentage']
   rating_product = data['stats']['rating']
   review_product = data['stats']['reviewCount']

   product = {
      'nama_product' : nama_product,
      'foto_product' : foto_product,
      'harga_product' : harga_product,
      'diskon_product' : diskon_product,
      'rating_product' : rating_product,
      'review_product' : review_product
   }
   daftar_product.append(product)

data_frame = pandas.DataFrame(daftar_product)
data_frame.to_csv('dataset.csv', encoding="utf-8")