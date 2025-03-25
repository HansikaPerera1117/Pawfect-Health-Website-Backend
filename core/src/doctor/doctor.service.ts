import { HttpException, HttpStatus, Injectable } from "@nestjs/common";
import { PrismaService } from "src/prisma/prisma.service";

@Injectable()
export class DoctorService {
  constructor(private readonly prismaService: PrismaService) {}
  
  async findNearestDoctor(data: { lat: number; lng: number }) {
    const { lat, lng } = data;

    try {
      const nearestDoctors = await this.prismaService.$queryRaw`
        SELECT *, 
            (6371 * acos(
                cos(radians(${lat})) * cos(radians(lat)) * 
                cos(radians(lng) - radians(${lng})) + 
                sin(radians(${lat})) * sin(radians(lat))
            )) AS distance
        FROM "Doctor"
        WHERE (6371 * acos(
                cos(radians(${lat})) * cos(radians(lat)) * 
                cos(radians(lng) - radians(${lng})) + 
                sin(radians(${lat})) * sin(radians(lat))
            )) <= 10
        ORDER BY distance ASC;
      `;

      return nearestDoctors;
    } catch (error) {
      console.error("Error fetching nearest doctors:", error);
      throw new Error("Could not fetch nearest doctors");
    }
  }

  create(data: any) {
    console.log(data);

    return this.prismaService.doctor.create({
      data: {
        email: data.email,
        address: data.address,
        contact: data.contact,
        password: data.password,
        lat: data.lat,
        lng: data.lng,
        userType: "DOCTOR",
        position: data.position,
        description: data.description,
        name: data.name,
      },
    });
  }

  findById(doctorId: string) {
    const dcId = parseInt(doctorId, 10);
    return this.prismaService.doctor.findUnique({
      where: {
        id: dcId,
      },
    });
  }

  async loginDoctor(email: any, password: any) {
    try {
      const isExsits = await this.prismaService.doctor.findUnique({
        where: {
          email: email,
        },
      });
      console.log(isExsits);

      if (isExsits && (await isExsits.password) === password) {
        return isExsits;
      } else {
        throw new HttpException(
          "Invalid email or password",
          HttpStatus.BAD_REQUEST
        );
      }
    } catch (error) {
      console.log(error);

      throw new HttpException(
        "Invalid email or password",
        HttpStatus.BAD_REQUEST
      );
    }
  }

}
