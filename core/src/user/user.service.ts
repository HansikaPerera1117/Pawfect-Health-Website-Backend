import { HttpException, HttpStatus, Injectable } from "@nestjs/common";
import { PrismaService } from "src/prisma/prisma.service";

@Injectable()
export class UserService {
  constructor(private readonly prismaService: PrismaService) {}

  async register(data: any) {
    
    const isExsits = await this.prismaService.user.findUnique({
      where: {
        email: data.email,
      },
    });

    if (isExsits) {
      throw new HttpException(
        `sorry! your enterd ${isExsits.email} is alredy exsits`,
        HttpStatus.BAD_REQUEST
      );
    }

    return this.prismaService.user.create({
      data: {
        firstName: data.firstName,
        lastName: data.lastName,
        password: data.password,
        email: data.email,
        userType: "DOG_OWNER",
        lat: data.lat,
        lng: data.lng,
      },
    });
  }


  async loginUser(email: string, password: string) {
    try {
      // Search for the user in both tables
      const user = await this.prismaService.user.findUnique({
        where: { email: email },
      });

      const doctor = await this.prismaService.doctor.findUnique({
        where: { email: email },
      });

      // Determine if the account exists
      const account = user || doctor;

      if (!account) {
        throw new HttpException(
          "Invalid email or password",
          HttpStatus.BAD_REQUEST
        );
      }

      if (account.password !== password) {
        throw new HttpException(
          "Invalid email or password",
          HttpStatus.BAD_REQUEST
        );
      }

      return account; // Return user or doctor details
    } catch (error) {
      console.error(error);
      throw new HttpException(
        "Invalid email or password",
        HttpStatus.BAD_REQUEST
      );
    }
  }
}
